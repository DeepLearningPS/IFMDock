"""Canonical receptor-file selection shared by physical evaluation tools."""

from pathlib import Path


def resolve_posebusters_protein(complex_dir: str | Path, complex_id: str) -> Path:
    """Select the PoseBusters conditioning receptor in EC-Dock priority order."""

    complex_dir = Path(complex_dir)
    candidates = (
        complex_dir / f"origin_{complex_id}_protein.pdb",
        complex_dir / f"{complex_id}_protein.pdb",
        complex_dir / f"{complex_id}_ref_protein.pdb",
    )
    for path in candidates:
        if path.is_file():
            return path
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"no PoseBusters conditioning protein for {complex_id}; tried: {attempted}"
    )
