import argparse
import shutil
from pathlib import Path


INTERFACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTERFACE_DIR.parents[2]


def resolve_interface_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else INTERFACE_DIR / path


def main():
    parser = argparse.ArgumentParser(
        description="Move distance-model interaction files into a prepared dataset."
    )
    parser.add_argument("--data_name", default="tmpdata")
    parser.add_argument("--source_dir", default=None)
    parser.add_argument(
        "--random_cutoff_dir",
        default="tmpdata_predict_sdf_random_protein_cutoff",
    )
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    if not args.data_name or "/" in args.data_name or "\\" in args.data_name:
        parser.error("--data_name must be one directory name, not a path")

    source_dir = args.source_dir or f"{args.data_name}_predict_sdf_boxsize10"
    source_path = resolve_interface_path(source_dir)
    project_root = Path(args.project_root).resolve()
    target_path = project_root / args.data_name / args.data_name
    count = 0

    if not source_path.is_dir():
        raise FileNotFoundError(f"Distance interaction directory not found: {source_path}")

    for item in source_path.iterdir():
        if item.is_dir() and any(item.iterdir()):
            source_file = item / f"interaction_{item.name}.pkl"
            target_file = target_path / item.name / f"interaction_{item.name}_v2.pkl"
            if not source_file.is_file():
                raise FileNotFoundError(f"Interaction file not found: {source_file}")
            shutil.copy2(source_file, target_file)
            count += 1

    print("success num:", count)

    if source_path.is_dir():
        shutil.rmtree(source_path)
        print(f" {source_path} ")

    random_cutoff_path = resolve_interface_path(args.random_cutoff_dir)
    if random_cutoff_path.is_dir():
        shutil.rmtree(random_cutoff_path)
        print(f" {random_cutoff_path} ")


if __name__ == "__main__":
    main()
