#!/usr/bin/env python3
import argparse, copy, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from ifmdock.data.feature.featurizer import FeaturizerConfig
from ifmdock.data.modules.inference import PredictionDataset
from ifmdock.data.transforms.docking.pocket import PocketTransform
from ifmdock.models.pl_modules.docking import load_pretrained_model
from ifmdock.sampling.docking.diffusion import set_time_t_dict
from ifmdock.metrics.relaxation import (
    compute_posebusters_geometry_metrics,
    compute_posebusters_interaction_metrics,
)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--initial-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True); p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1); p.add_argument('--steps',type=int,default=15)
    p.add_argument('--pose-batch-size',type=int,default=8); p.add_argument('--limit',type=int,default=0)
    p.add_argument(
        '--distance-candidate-selection',
        choices=('model_default','single','random','balanced_random','oracle_best'),
        default='model_default',
    )
    p.add_argument(
        '--quick-physical-fallback', action='store_true',
        help=(
            'Opt in to the legacy behavior: restore the Stage-1 pose when the '
            'Stage-2 quick geometry/contact diagnostic fails.  By default the '
            'diagnostic is logged only and the Stage-2 pose is retained.'
        ),
    )
    # Compatibility with older launchers.  The default is already no fallback.
    p.add_argument('--no-physical-fallback', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--rdkit-initial-pose-root', type=Path, default=None)
    a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True)
    if a.quick_physical_fallback and a.no_physical_fallback:
        p.error('--quick-physical-fallback and --no-physical-fallback cannot be used together')
    cfg=OmegaConf.load(a.model_dir/'model_parameters.yml')
    if a.distance_candidate_selection != 'model_default':
        cfg.model.distance_guidance.cached_candidate_selection = a.distance_candidate_selection
    model=load_pretrained_model(cfg,a.checkpoint,use_ema_weights=True,freeze=True).model.cuda().eval()
    frame=pd.read_csv(a.inputs)
    if a.limit: frame=frame.iloc[:a.limit]
    indices=list(range(a.shard_index,len(frame),a.num_shards))
    featurizer_cfg=FeaturizerConfig(
        matching=False, popsize=None, maxiter=None, keep_original=False,
        remove_hs=True, num_conformers=1, max_lig_size=None,
        flexible_backbone=False, flexible_sidechains=False, rigid_pocket=True,
    )
    ds=PredictionDataset(str(a.inputs),featurizer_cfg,rigid_pocket=True)
    pocket=PocketTransform(pocket_reduction=False,all_atoms=True,flexible_backbone=False,flexible_sidechains=False)
    done=failed=0
    for seq,idx in enumerate(indices,1):
        name=str(frame.iloc[idx].pdbid); src=a.initial_root/name/'docking_predictions.pkl'; out=a.output_root/name/'docking_predictions.pkl'
        if out.exists(): done+=1; continue
        try:
            payload=pickle.load(open(src,'rb')); initial=[np.asarray(x,dtype=np.float32) for x in payload['ligand_pos']]
            base=ds.get(idx)
            base['ligand'].ecdock_reference_pos=base['ligand'].pos.clone(); base['atom'].ecdock_reference_pos=base['atom'].pos.clone()
            if a.rdkit_initial_pose_root is not None:
                sidecar = a.rdkit_initial_pose_root / f'{name}.npy'
                if sidecar.is_file():
                    base['ligand'].ecdock_reference_pos = torch.from_numpy(
                        np.asarray(np.load(sidecar), dtype=np.float32)
                    ).clone()
            base['ligand'].orig_pos=base['ligand'].pos.clone(); base=pocket(base)
            center=base.original_center.reshape(1,3)
            refined=[]; fallback_mask=[]; quick_failure_mask=[]
            for start in range(0,len(initial),a.pose_batch_size):
                graphs=[]
                for local_index, xyz in enumerate(initial[start:start+a.pose_batch_size]):
                    g=copy.deepcopy(base); g['ligand'].pos=torch.from_numpy(xyz)-center
                    # Match Stage 1's pose-wise distance candidate exactly.
                    g.distance_guidance_candidate_slot = start + local_index
                    graphs.append(g)
                b=Batch.from_data_list(graphs).cuda()
                dt=1.0/a.steps
                for step in range(a.steps):
                    t=step/a.steps; td={k:t for k in ('tr','rot','tor','t')}; td.update(sc_tor=None,bb_tr=None,bb_rot=None)
                    set_time_t_dict(b,td,b.num_graphs,True,device=b['ligand'].pos.device)
                    with torch.no_grad(): v=model(b,fast_updates=True)['ligand_velocity']
                    b['ligand'].pos=b['ligand'].pos+dt*v
                centers=b.original_center.reshape(b.num_graphs,3)[b['ligand'].batch]
                absolute=(b['ligand'].pos+centers).cpu()
                ptr=b['ligand'].ptr.cpu()
                batch_refined=[absolute[ptr[j]:ptr[j+1]].numpy() for j in range(b.num_graphs)]
                # The inexpensive checks are retained as diagnostics.  They are
                # deliberately not a hard replacement rule by default: the raw
                # Stage-2 output has been empirically more accurate and more
                # PoseBusters-valid than the legacy quick-rule rollback.
                lig_dense,_=to_dense_batch(b['ligand'].pos,b['ligand'].batch)
                atom_dense,_=to_dense_batch(b['atom'].pos,b['atom'].batch)
                edge_store=base['ligand','lig_edge','ligand']
                device=lig_dense.device
                checks={}
                checks.update(compute_posebusters_geometry_metrics(
                    lig_dense, edge_store.posebusters_edge_index.to(device),
                    edge_store.lower_bound.to(device), edge_store.upper_bound.to(device),
                    edge_store.posebusters_bond_mask.to(device),
                    edge_store.posebusters_angle_mask.to(device),
                ))
                checks.update(compute_posebusters_interaction_metrics(
                    lig_dense, atom_dense, base['ligand'].vdw_radii.to(device),
                    base['atom'].vdw_radii.to(device),
                ))
                passes=np.all(np.stack(list(checks.values())),axis=0)
                starts=initial[start:start+a.pose_batch_size]
                for local_idx,(candidate,passed) in enumerate(zip(batch_refined,passes)):
                    quick_failed=not bool(passed)
                    use_fallback=bool(a.quick_physical_fallback and quick_failed)
                    refined.append(starts[local_idx] if use_fallback else candidate)
                    quick_failure_mask.append(quick_failed)
                    fallback_mask.append(use_fallback)
            result=dict(payload); result['ligand_pos']=refined; result['stage2_steps']=a.steps
            result['stage2_quick_physical_failure_mask']=quick_failure_mask
            result['stage2_quick_physical_failure_count']=int(sum(quick_failure_mask))
            result['stage2_physical_fallback_enabled']=bool(a.quick_physical_fallback)
            result['stage2_physical_fallback_mask']=fallback_mask
            result['stage2_physical_fallback_count']=int(sum(fallback_mask))
            out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix('.pkl.tmp')
            with open(tmp,'wb') as h: pickle.dump(result,h)
            tmp.replace(out); done+=1
        except Exception as e:
            failed+=1; print(f'{name} FAILED {type(e).__name__}: {e}',flush=True)
        print(f'shard={a.shard_index} {seq}/{len(indices)} ok={done} failed={failed} {name}',flush=True)

if __name__=='__main__': main()
