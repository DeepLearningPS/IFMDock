from pathlib import Path
import importlib
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("generalmodels.tasks." + file.name[:-3])
