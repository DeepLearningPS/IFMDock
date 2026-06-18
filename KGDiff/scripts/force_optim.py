import torch
from .MMFF import MMFF_Energy, combine_coords_with_masks, combine_MMFF_params_from_mols,combine_fix_masks_with_masks, MMFF_keys, MMFF_pad_dim
from ..comparm import GP

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)

class CoorMin(torch.nn.Module):
    def __init__(self):
        super(CoorMin, self).__init__()
        self.MMFF_lossfunction = MMFF_Energy(split_interact=False, warm=False)
        self.lr = GP.MMFF_lr
        self.decay = GP.MMFF_decay
        self.max_decay_step = GP.MMFF_max_decay_step
        self.patience_tol_step = GP.MMFF_patience_tol_step
        self.patience_tol_value = GP.MMFF_patience_tol_value
        self.clip = GP.MMFF_clip
        self.constraint = GP.MMFF_constraint

    def forward(self, l_coor_pred, mmff_params, fix_masks, loop=None, constraint=None, show_state=False, min_type='GD',return_delta=False,):
        with TemporaryGrad():  # i.e. torch.set_grad_enabled(True)
            coor_min,energy_min = self.FF_min(l_coor_pred, mmff_params,fix_masks=fix_masks,
               loop=self.loop if isinstance(loop, type(None)) else loop,
               lr=self.lr,
               decay=self.decay, max_decay_step=self.max_decay_step,
               patience_tol_step=self.patience_tol_step, patience_tol_value=self.patience_tol_value,
               clip=self.clip,
               constraint=self.constraint if isinstance(constraint, type(None)) else constraint,
               show_state=show_state, min_type=min_type,
               )
        if torch.isnan(coor_min).any():
            if return_delta:
                return torch.zeros_like(l_coor_pred).to(l_coor_pred.device),energy_min
            else:
                return l_coor_pred,energy_min
        else:
            if return_delta:
                return coor_min - l_coor_pred, energy_min 
            else:
                return coor_min,energy_min 

    def FF_min(self, coor_pred, mmff_params,fix_masks=None,
               loop=10000, lr=5e-5,
               decay=0.5, max_decay_step=10,
               patience_tol_step=10, patience_tol_value=0,
               clip=1e+5, constraint=0, show_state=False, min_type='GD',):

        coor_pred_detach = coor_pred.detach()
        coor_paramed = torch.nn.Parameter(coor_pred.detach().clone()).to(coor_pred.device)
        if min_type == 'GD':
            optimizer = torch.optim.SGD([coor_paramed], lr=lr)
        elif min_type == 'LBFGS':
            optimizer = torch.optim.LBFGS([coor_paramed], lr=1.0, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay)

        for step in range(loop):
            def closure():
                e = self.MMFF_lossfunction(coor_paramed, mmff_params, return_sum=True)
                e = e.sum()
                if constraint > 0:
                    e_constraint = 0.5 * ((coor_paramed - coor_pred_detach) ** 2 * constraint).sum()
                    e = e + e_constraint
                optimizer.zero_grad()
                e.backward()
                if min_type == 'GD':
                    torch.nn.utils.clip_grad_norm_([coor_paramed], clip)
                return e
            e = closure()

            if min_type == 'GD':
                optimizer.step()
            elif min_type == 'LBFGS':
                optimizer.step(closure)
            coord_paramed = coor_pred_detach*fix_masks+coor_paramed*(1-fix_masks)
            if step == 0:
                e_min = e
                patience_sum = 0
                decay_step = 0
            elif e_min - e < patience_tol_value:
                patience_sum += 1
            else:
                e_min = e
            if patience_sum >= patience_tol_step:
                if decay_step < max_decay_step:
                    scheduler.step()
                    patience_sum = 0
                decay_step += 1
            if torch.isnan(e).any():
                break

            if show_state:
                print(f"Step:{step + 1}, lr:{optimizer.param_groups[0]['lr']:.2e}, E:{e.detach().cpu().numpy():.3e}")
                #print (e_split)
        coor_pred_min = coor_paramed.detach().clone().to(coor_pred.device)
        energy_min=self.MMFF_lossfunction(coor_pred_min, mmff_params, return_sum=True)
        del optimizer
        del scheduler
        del coor_paramed
        torch.cuda.empty_cache()
        return coor_pred_min,energy_min
    
def opt_coords_moves(coords_pred,mols,coords_masks,fix_masks=None,loop=1,show_state=False,min_type='LBFGS',):
    coords_shape=coords_pred.shape
    coords_masks_=coords_masks.view(-1)
    coords_selected=combine_coords_with_masks(coords_pred,coords_masks)
    coords_fix_masks=combine_fix_masks_with_masks(fix_masks,coords_masks)

    mmff_params=combine_MMFF_params_from_mols(mols)

    for key in mmff_params.keys():
        mmff_params[key]=mmff_params[key].to(coords_pred.device)
    coord_opt=CoorMin().to(coords_pred.device)
    coord_moves,energy_min=coord_opt(coords_selected,mmff_params,fix_masks=coords_fix_masks,loop=loop,show_state=show_state,min_type=min_type,return_delta=True,)
    coords_moves_=torch.zeros(coords_shape).to(coords_pred.device)
    coords_moves_=coords_moves_.view(-1,3)
    coords_moves_[coords_masks_]=coord_moves
    coords_moves_=coords_moves_.view(*coords_shape)
    return coords_moves_,energy_min
