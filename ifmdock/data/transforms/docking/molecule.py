import numpy as np
import torch

from ifmdock.geometry.manifolds import so3, torus
from ifmdock.geometry.ops import quaternion_to_axis_angle
from ifmdock.data.conformers.modify import modify_conformer
from ifmdock.geometry.torsion import canonical_dihedrals, periodic_wrap


class LigandTransform:
    def __init__(
        self,
        no_torsion: bool = False,
        fast_updates: bool = False,
        transform_type: str = "diffusion",
        flow_tr_sigma: float = 30.0,
        flow_rot_sigma: float = 1.65,
        flow_tor_sigma: float = 3.14,
        flow_box_padding: float = 10.0,
        flow_source_mode: str = "matched_endpoint_noise",
        flow_rot_prior: str = "gaussian",
        flow_tor_prior: str = "gaussian",
        flow_tor_uniform_prob: float = 0.25,
        flow_tor_empirical_prob: float = 0.5,
        torsion_target_mode: str = "relative_velocity",
    ):
        self.no_torsion = no_torsion
        self.fast_updates = fast_updates
        self.transform_type = transform_type
        self.flow_tr_sigma = flow_tr_sigma
        self.flow_rot_sigma = flow_rot_sigma
        self.flow_tor_sigma = flow_tor_sigma
        self.flow_box_padding = flow_box_padding
        self.flow_source_mode = flow_source_mode
        self.flow_rot_prior = flow_rot_prior
        self.flow_tor_prior = flow_tor_prior
        self.flow_tor_uniform_prob = float(flow_tor_uniform_prob)
        self.flow_tor_empirical_prob = float(flow_tor_empirical_prob)
        if not 0.0 <= self.flow_tor_uniform_prob <= 1.0:
            raise ValueError("flow_tor_uniform_prob must be in [0, 1]")
        if not 0.0 <= self.flow_tor_empirical_prob <= 1.0:
            raise ValueError("flow_tor_empirical_prob must be in [0, 1]")
        self.torsion_target_mode = torsion_target_mode

    def __call__(self, data, t_dict, sigma_dict):
        if self.transform_type == "flow":
            return self.apply_flow_transform(data, t_dict)
        return self.apply_diffusion_transform(data, t_dict, sigma_dict)

    def apply_flow_transform(self, data, t_dict):
        """Interpolate a normal pose prior at t=0 to crystal pose at t=1."""
        if not hasattr(data["ligand"], "orig_pos"):
            data["ligand"].orig_pos = data["ligand"].pos.clone()

        t = float(t_dict["t"])
        ligand_center = data["ligand"].pos.mean(dim=0, keepdim=True)
        # PocketTransform has already centered atom.pos.  The immutable apo
        # tensor is deliberately not used here because rigid-pocket mode does
        # not center it.
        atom_pos = data["atom"].pos
        pocket_center = atom_pos.mean(dim=0, keepdim=True)
        if self.flow_tr_sigma > 0:
            tr_std = ligand_center.new_full((1, 3), self.flow_tr_sigma)
        else:
            # The dataset docking box is the native ligand bounding box plus
            # 10 A.  N(0, (box/6)^2) keeps 99.7% of centers within half-box.
            ligand_extent = data["ligand"].pos.amax(0) - data["ligand"].pos.amin(0)
            tr_std = ((ligand_extent + self.flow_box_padding) / 6.0).unsqueeze(0)
        tr_prior = pocket_center - ligand_center + torch.randn_like(tr_std) * tr_std
        if self.flow_rot_prior == "haar" or self.flow_source_mode == "rdkit_raw_pair_haar":
            # Shoemake's construction: a uniform unit quaternion, hence Haar
            # measure on SO(3).  Use the principal representative (w >= 0) so
            # scaling the log vector below is the shortest geodesic.
            u = torch.rand(3, dtype=ligand_center.dtype, device=ligand_center.device)
            q_xyzw = torch.stack(
                (
                    torch.sqrt(1 - u[0]) * torch.sin(2 * torch.pi * u[1]),
                    torch.sqrt(1 - u[0]) * torch.cos(2 * torch.pi * u[1]),
                    torch.sqrt(u[0]) * torch.sin(2 * torch.pi * u[2]),
                    torch.sqrt(u[0]) * torch.cos(2 * torch.pi * u[2]),
                )
            )
            quaternion = torch.cat((q_xyzw[3:], q_xyzw[:3])).unsqueeze(0)
            quaternion = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)
            rot_prior = quaternion_to_axis_angle(quaternion)
        else:
            rot_prior = torch.normal(mean=0.0, std=self.flow_rot_sigma, size=(1, 3))
        # Match Matcha's SO(3) logarithm target: use the principal (shortest)
        # axis-angle representative even when a normal draw exceeds pi.
        rot_angle = torch.linalg.vector_norm(rot_prior, dim=-1, keepdim=True)
        principal_angle = torch.remainder(rot_angle + np.pi, 2 * np.pi) - np.pi
        rot_prior = rot_prior * (principal_angle / rot_angle.clamp_min(1e-8))

        n_torsions = int(data["ligand"].edge_mask.sum())
        if self.no_torsion:
            torsion_prior = np.empty(0, dtype=np.float32)
        else:
            if self.flow_tor_prior == "uniform":
                torsion_prior = np.random.uniform(
                    low=-np.pi, high=np.pi, size=n_torsions
                ).astype(np.float32)
            elif self.flow_tor_prior == "mixed":
                gaussian = np.random.normal(
                    loc=0.0, scale=self.flow_tor_sigma, size=n_torsions
                ).astype(np.float32)
                uniform = np.random.uniform(
                    low=-np.pi, high=np.pi, size=n_torsions
                ).astype(np.float32)
                use_uniform = np.random.random(n_torsions) < self.flow_tor_uniform_prob
                torsion_prior = np.where(use_uniform, uniform, gaussian)
            else:
                torsion_prior = np.random.normal(
                    loc=0.0, scale=self.flow_tor_sigma, size=n_torsions
                ).astype(np.float32)
            if self.flow_source_mode in {
                "rdkit_matched_pair", "rdkit_matched_noise", "rdkit_raw_pair_haar"
            }:
                if not hasattr(data["ligand"], "rdkit_source_tor_offset"):
                    raise ValueError(
                        f"{self.flow_source_mode} requires a conformer-matched training cache"
                    )
            use_empirical = False
            if self.flow_source_mode in {"rdkit_matched_pair", "rdkit_raw_pair_haar"} or self.flow_tor_prior == "empirical_mixed":
                base_offset = np.asarray(
                    data["ligand"].rdkit_source_tor_offset, dtype=np.float32
                )
                if base_offset.shape != torsion_prior.shape:
                    raise ValueError("RDKit source torsion offset shape mismatch")
                if self.flow_tor_prior == "empirical_mixed":
                    use_empirical = bool(np.random.random() < self.flow_tor_empirical_prob)
                    if use_empirical:
                        torsion_prior = base_offset.copy()
                elif self.flow_source_mode == "rdkit_raw_pair_haar":
                    torsion_prior = base_offset
                else:
                    torsion_prior = torsion_prior + base_offset
            # rdkit_matched_noise deliberately keeps the matched RDKit graph
            # as the source geometry and adds only the configured stochastic
            # torsion prior. This is the exact source used by matched_paired
            # inference before flow integration.
            if self.flow_source_mode == "rdkit_raw_pair_haar" or use_empirical:
                periods = np.asarray(
                    getattr(data["ligand"], "rdkit_source_tor_period",
                            np.full(n_torsions, 2 * np.pi)),
                    dtype=np.float32,
                )
                if periods.shape != torsion_prior.shape:
                    raise ValueError("RDKit source torsion period shape mismatch")
                torsion_prior = (torsion_prior + periods / 2) % periods - periods / 2
            else:
                torsion_prior = (torsion_prior + np.pi) % (2 * np.pi) - np.pi

        endpoint_torsion = None
        canonical_available = (
            hasattr(data["ligand"], "canonical_torsion_index")
            and hasattr(data["ligand"], "canonical_tor_period")
        )
        if canonical_available:
            index = data["ligand"].canonical_torsion_index
            endpoint_torsion = canonical_dihedrals(data["ligand"].pos, index)

        remaining = 1.0 - t
        modify_conformer(
            data,
            tr_prior * remaining,
            rot_prior.squeeze(0) * remaining,
            torsion_prior * remaining,
            fast=self.fast_updates,
        )
        data.tr_flow = -tr_prior
        data.rot_flow = -rot_prior
        if canonical_available:
            current_torsion = canonical_dihedrals(data["ligand"].pos, index)
            periods = data["ligand"].canonical_tor_period.to(current_torsion)
            data["ligand"].tor_current = current_torsion
            data["ligand"].tor_endpoint = torch.stack(
                (torch.cos(endpoint_torsion), torch.sin(endpoint_torsion)), dim=-1
            )
        if self.torsion_target_mode in {"canonical_velocity", "canonical_endpoint"}:
            canonical_velocity = periodic_wrap(endpoint_torsion - current_torsion, periods) / max(remaining, 1e-6)
            data.tor_flow = canonical_velocity
        else:
            data.tor_flow = torch.from_numpy(-torsion_prior).float()
        return data

    def apply_diffusion_transform(self, data, t_dict, sigma_dict):
        # Preserve the crystallographic pose before applying translation,
        # rotation and torsional noise.  Reference-distance guidance uses this
        # immutable tensor to build a static EC-Dock contact graph.
        if not hasattr(data["ligand"], "orig_pos"):
            data["ligand"].orig_pos = data["ligand"].pos.clone()
        tr_sigma, rot_sigma = sigma_dict["tr_sigma"], sigma_dict["rot_sigma"]
        if not self.no_torsion:
            tor_sigma = sigma_dict["tor_sigma"]

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3))
        rot_update = so3.sample_vec(eps=rot_sigma)

        if not self.no_torsion:
            torsion_updates = np.random.normal(
                loc=0.0, scale=tor_sigma, size=data["ligand"].edge_mask.sum()
            )

        # Now modified to move between fast (on GPU) vs CPU <-> GPU
        modify_conformer(
            data,
            tr_update,
            torch.from_numpy(rot_update).float(),
            torsion_updates,
            fast=self.fast_updates,
        )

        data.tr_score = -tr_update / tr_sigma**2
        data.rot_score = (
            torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma))
            .float()
            .unsqueeze(0)
        )
        data.tor_score = (
            None
            if self.no_torsion
            else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        )
        data.tor_sigma_edge = (
            None
            if self.no_torsion
            else np.ones(data["ligand"].edge_mask.sum()) * tor_sigma
        )

        return data
