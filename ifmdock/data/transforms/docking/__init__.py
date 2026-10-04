from functools import partial

import numpy as np
from torch_geometric.transforms import BaseTransform, Compose

from ifmdock.data.transforms.docking.molecule import LigandTransform
from ifmdock.data.transforms.docking.pocket import PocketTransform, UnbalancedTransform
from ifmdock.data.transforms.docking.protein import (
    ProteinTransform,
    NearbyAtomsTransform,
)
from ifmdock.data.transforms.docking.bb_priors import construct_bb_prior
from ifmdock.data.transforms.docking.cartesian_flow import CartesianLigandFlowTransform

from ifmdock.sampling.docking.diffusion import (
    t_to_sigma as t_to_sigma_compl,
    set_time_t_dict,
    bridge_transform_t,
)
from ifmdock.utils.configs import TimeConfig, SigmaConfig


class DockingTransform(BaseTransform):
    """Transform class for docking.

    Samples t, and applies updates to protein and ligand.
    """

    def __init__(
        self,
        time_config: TimeConfig,
        sigma_config: SigmaConfig,
        lig_transform: LigandTransform,
        prot_transform: ProteinTransform,
        all_atoms: bool,
        include_miscellaneous_atoms: bool = False,
    ):
        self.all_atoms = all_atoms
        self.include_miscellaneous_atoms = include_miscellaneous_atoms

        if time_config.sc_tor_bridge_alpha is None:
            assert sigma_config.sidechain_tor_sigma is None

        if time_config.bb_rot_bridge_alpha is None:
            assert sigma_config.bb_rot_sigma is None
            assert sigma_config.bb_tr_sigma is None

        self.t_to_sigma = partial(t_to_sigma_compl, args=sigma_config)

        self.time_config = time_config
        self.lig_transform = lig_transform
        self.prot_transform = prot_transform

    def sample_t(self, data):
        t_lig = np.random.beta(
            self.time_config.sampling_alpha, self.time_config.sampling_beta
        )
        t_dict = {}

        t_dict["tr"], t_dict["rot"], t_dict["tor"] = t_lig, t_lig, t_lig
        t_dict["t"] = t_lig

        if self.time_config.sc_tor_bridge_alpha is not None:
            t_sc_tor = bridge_transform_t(
                t=t_lig, alpha=self.time_config.sc_tor_bridge_alpha
            )
        else:
            t_sc_tor = None

        if self.time_config.bb_rot_bridge_alpha is not None:
            t_bb_tr = bridge_transform_t(
                t=t_lig, alpha=self.time_config.bb_tr_bridge_alpha
            )
            t_bb_rot = bridge_transform_t(
                t=t_lig, alpha=self.time_config.bb_rot_bridge_alpha
            )
        else:
            t_bb_tr, t_bb_rot = None, None

        t_dict["sc_tor"] = t_sc_tor
        t_dict["bb_tr"], t_dict["bb_rot"] = t_bb_tr, t_bb_rot

        set_time_t_dict(
            data,
            t_dict,
            1,
            self.all_atoms,
            device=None,
            include_miscellaneous_atoms=self.include_miscellaneous_atoms,
        )

        return t_dict

    def apply_transform(self, data, t_dict, sigma_dict):
        data = self.lig_transform(data, t_dict, sigma_dict)
        data = self.prot_transform(data, t_dict, sigma_dict)
        return data

    def __call__(self, data):
        t_dict = self.sample_t(data)
        sigma_dict = self.t_to_sigma(t_dict)
        return self.apply_transform(data, t_dict, sigma_dict)


def construct_transform(cfg, mode="train"):
    if cfg.get("rigid_pocket", False) and (
        cfg.flexible_backbone or cfg.flexible_sidechains
    ):
        raise ValueError(
            "rigid_pocket=True requires flexible_backbone=False and "
            "flexible_sidechains=False."
        )
    transforms = []

    pocket_transform = PocketTransform(
        pocket_reduction=cfg.pocket.pocket_reduction,
        pocket_buffer=cfg.pocket.pocket_buffer,
        all_atoms=cfg.pocket.all_atoms,
        flexible_backbone=cfg.flexible_backbone,
        flexible_sidechains=cfg.flexible_sidechains,
        fast_updates=cfg.fast_updates,
    )
    transforms.append(pocket_transform)

    if mode in ["train", "val"] and cfg.get("cartesian_flow", False):
        transforms.append(
            CartesianLigandFlowTransform(
                alpha=cfg.time_args.sampling_alpha,
                beta=cfg.time_args.sampling_beta,
                all_atoms=cfg.pocket.all_atoms,
                source_mode=cfg.get("cartesian_source_mode", "stage2_pose"),
                gaussian_sigma=cfg.get("cartesian_gaussian_sigma", 3.0),
                rdkit_translation_sigma=cfg.get(
                    "cartesian_rdkit_translation_sigma", 3.0
                ),
                rdkit_random_rotation=cfg.get(
                    "cartesian_rdkit_random_rotation", True
                ),
            )
        )
    elif mode in ["train", "val"]:
        unbalanced_transform = UnbalancedTransform(
            match_max_rmsd=cfg.unbalanced.match_max_rmsd if mode == "train" else None,
            fast_updates=cfg.fast_updates,
        )
        transforms.append(unbalanced_transform)

    nearby_atom_transform = NearbyAtomsTransform(
        only_nearby_residues_atomic=cfg.nearby_atoms.only_nearby_residues_atomic,
        nearby_residues_atomic_radius=cfg.nearby_atoms.nearby_residues_atomic_radius,
        nearby_residues_atomic_min=cfg.nearby_atoms.nearby_residues_atomic_min,
        fast_updates=cfg.fast_updates,
    )
    transforms.append(nearby_atom_transform)

    # Cartesian stage 2 already sampled its sole time and constructed
    # x_t=(1-t)x_0+t*x_1 above.  Appending DockingTransform here would sample
    # a second time and apply the legacy translation/rotation/torsion noise,
    # breaking the correspondence between x_t, t and the velocity target.
    if mode in ["train", "val"] and not cfg.get("cartesian_flow", False):
        time_config = TimeConfig(
            sampling_alpha=cfg.time_args.sampling_alpha,
            sampling_beta=cfg.time_args.sampling_beta,
            bb_tr_bridge_alpha=cfg.time_args.bb_tr_bridge_alpha
            if cfg.flexible_backbone
            else None,
            bb_rot_bridge_alpha=cfg.time_args.bb_rot_bridge_alpha
            if cfg.flexible_backbone
            else None,
            sc_tor_bridge_alpha=cfg.time_args.sc_tor_bridge_alpha
            if cfg.flexible_sidechains
            else None,
        )

        sigma_config = SigmaConfig(
            tr_sigma_max=cfg.sigma_args.tr_sigma_max,
            tr_sigma_min=cfg.sigma_args.tr_sigma_min,
            rot_sigma_max=cfg.sigma_args.rot_sigma_max,
            rot_sigma_min=cfg.sigma_args.rot_sigma_min,
            tor_sigma_max=cfg.sigma_args.tor_sigma_max,
            tor_sigma_min=cfg.sigma_args.tor_sigma_min,
            bb_rot_sigma=cfg.sigma_args.bb_rot_sigma if cfg.flexible_backbone else None,
            bb_tr_sigma=cfg.sigma_args.bb_tr_sigma if cfg.flexible_backbone else None,
            sidechain_tor_sigma=cfg.sigma_args.sidechain_tor_sigma
            if cfg.flexible_sidechains
            else None,
        )

        lig_transform = LigandTransform(
            no_torsion=cfg.ligand.no_torsion,
            fast_updates=cfg.fast_updates,
            transform_type=cfg.ligand.get("transform_type", "diffusion"),
            flow_tr_sigma=cfg.ligand.get("flow_tr_sigma", cfg.sigma_args.tr_sigma_max),
            flow_rot_sigma=cfg.ligand.get("flow_rot_sigma", cfg.sigma_args.rot_sigma_max),
            flow_tor_sigma=cfg.ligand.get("flow_tor_sigma", cfg.sigma_args.tor_sigma_max),
            flow_box_padding=cfg.ligand.get("flow_box_padding", 10.0),
            flow_source_mode=cfg.ligand.get(
                "flow_source_mode", "matched_endpoint_noise"
            ),
            flow_rot_prior=cfg.ligand.get("flow_rot_prior", "gaussian"),
            flow_tor_prior=cfg.ligand.get("flow_tor_prior", "gaussian"),
            flow_tor_uniform_prob=cfg.ligand.get("flow_tor_uniform_prob", 0.25),
            flow_tor_empirical_prob=cfg.ligand.get("flow_tor_empirical_prob", 0.5),
            torsion_target_mode=cfg.ligand.get("torsion_target_mode", "relative_velocity"),
        )

        prot_transform = ProteinTransform(
            flexible_backbone=cfg.flexible_backbone,
            flexible_sidechains=cfg.flexible_sidechains,
            sidechain_tor_bridge=cfg.protein.sidechain_tor_bridge,
            use_bb_orientation_feats=cfg.protein.use_bb_orientation_feats,
            bb_prior=(
                construct_bb_prior(cfg.bb_prior)
                if cfg.flexible_backbone
                else None
            ),
            fast_updates=cfg.fast_updates,
        )

        docking_transform = DockingTransform(
            sigma_config=sigma_config,
            time_config=time_config,
            lig_transform=lig_transform,
            prot_transform=prot_transform,
            all_atoms=cfg.pocket.all_atoms,
            include_miscellaneous_atoms=False,
        )
        transforms.append(docking_transform)

    transform = Compose(transforms=transforms)
    return transform
