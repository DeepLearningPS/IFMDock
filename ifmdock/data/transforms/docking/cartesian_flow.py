import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

from ifmdock.sampling.docking.diffusion import set_time_t_dict


class CartesianLigandFlowTransform(BaseTransform):
    """Straight Cartesian flow from a configurable prior to the crystal pose.

    Only ligand Cartesian coordinates move. Protein atom/residue coordinates
    remain the rigid UniMol pocket throughout the trajectory.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, all_atoms=True,
                 source_mode: str = "stage2_pose", gaussian_sigma: float = 3.0,
                 rdkit_translation_sigma: float = 3.0,
                 rdkit_random_rotation: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.all_atoms = all_atoms
        self.source_mode = str(source_mode).lower()
        self.gaussian_sigma = float(gaussian_sigma)
        self.rdkit_translation_sigma = float(rdkit_translation_sigma)
        self.rdkit_random_rotation = bool(rdkit_random_rotation)

    @staticmethod
    def _random_rotation(dtype, device):
        matrix = torch.randn((3, 3), dtype=dtype, device=device)
        q, r = torch.linalg.qr(matrix)
        q = q @ torch.diag(torch.sign(torch.diag(r)))
        if torch.linalg.det(q) < 0:
            q[:, 0] *= -1
        return q

    def _initial_coordinates(self, data, target):
        pocket_center = data["receptor"].pos.to(target).mean(dim=0, keepdim=True)
        if self.source_mode == "stage2_pose":
            if not hasattr(data["ligand"], "stage2_initial_pos"):
                raise ValueError("source_mode=stage2_pose requires ligand.stage2_initial_pos")
            # Stage-1 sampling serializes poses in the original protein frame,
            # while PocketTransform has already centered the training graph.
            # Match the centered target (orig_pos - original_center) before
            # forming the Cartesian flow path; otherwise the target velocity
            # contains the full absolute pocket centroid as a spurious shift.
            center = torch.as_tensor(
                data.original_center, dtype=target.dtype, device=target.device
            ).reshape(1, 3)
            return data["ligand"].stage2_initial_pos.to(target) - center
        if self.source_mode == "rdkit":
            if not hasattr(data["ligand"], "rdkit_source_pos"):
                raise ValueError("source_mode=rdkit requires ligand.rdkit_source_pos")
            initial = data["ligand"].rdkit_source_pos.to(target).clone()
            initial -= initial.mean(dim=0, keepdim=True)
            if self.rdkit_random_rotation:
                initial = initial @ self._random_rotation(target.dtype, target.device).T
            shift = torch.randn((1, 3), dtype=target.dtype, device=target.device)
            return initial + pocket_center + self.rdkit_translation_sigma * shift
        if self.source_mode == "gaussian":
            return pocket_center + self.gaussian_sigma * torch.randn_like(target)
        raise ValueError(
            f"Unknown Cartesian source_mode={self.source_mode!r}; expected rdkit, gaussian, or stage2_pose"
        )

    def __call__(self, data):
        t = float(np.random.beta(self.alpha, self.beta))
        target = torch.as_tensor(
            data["ligand"].orig_pos,
            dtype=data["ligand"].pos.dtype,
            device=data["ligand"].pos.device,
        ).clone() - data.original_center.to(data["ligand"].pos).reshape(1, 3)
        initial = self._initial_coordinates(data, target)
        velocity = target - initial
        data["ligand"].stage2_initial_pos = initial
        data["ligand"].stage2_target_pos = target
        data["ligand"].cartesian_velocity_target = velocity
        data["ligand"].pos = initial + t * velocity
        # Keep the scalar explicitly for consistency checks and future
        # endpoint-based physical losses.  This must be the *only* time draw
        # and the ligand coordinates must not be perturbed again downstream.
        data.cartesian_flow_t = torch.tensor(t, dtype=target.dtype)
        t_dict = {k: t for k in ("tr", "rot", "tor", "t")}
        t_dict.update({"sc_tor": None, "bb_tr": None, "bb_rot": None})
        set_time_t_dict(data, t_dict, 1, self.all_atoms, device=None)

        expected = initial + t * velocity
        if not torch.allclose(data["ligand"].pos, expected, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Cartesian flow coordinate/time consistency check failed")
        return data
