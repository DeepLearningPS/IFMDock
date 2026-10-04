import torch


def canonical_dihedrals(pos: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Stable signed dihedrals for a canonical [4, N] atom-index tensor."""
    if index.numel() == 0:
        return pos.new_empty(0)
    p0, p1, p2, p3 = (pos.index_select(-2, index[k]) for k in range(4))
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = torch.cross(b0, b1, dim=-1)
    n1 = torch.cross(b1, b2, dim=-1)
    b1hat = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    m0 = torch.cross(n0, b1hat, dim=-1)
    return -torch.atan2((m0 * n1).sum(-1), (n0 * n1).sum(-1))


def periodic_wrap(delta: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    return torch.remainder(delta + period / 2, period) - period / 2
