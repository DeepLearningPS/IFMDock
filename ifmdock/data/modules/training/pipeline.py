import os
import dataclasses
import json
import signal
import traceback
from joblib import Parallel, delayed
import pickle
import logging

import torch

from ifmdock.data.parse.base import read_strings_from_txt
from ifmdock.data.parse.parser import ComplexParser
from ifmdock.data.feature.featurizer import Featurizer


class ComplexProcessingTimeout(TimeoutError):
    """Raised inside a worker when one complex exceeds its wall-clock budget."""


def _complex_name(complex_inputs):
    if isinstance(complex_inputs, dict):
        return str(complex_inputs.get("name", "unknown"))
    return str(getattr(complex_inputs, "name", "unknown"))


def _safe_featurize(featurizer, complex_inputs, timeout_seconds):
    """Convert every per-complex error into a result record.

    Joblib otherwise propagates a single RDKit/PyG exception to the parent and
    aborts a complete 1,000-complex shard.  SIGALRM is process-local on Linux,
    so it safely bounds pathological RDKit operations inside its own worker.
    """
    name = _complex_name(complex_inputs)
    previous_handler = None
    try:
        if timeout_seconds and timeout_seconds > 0:
            def _alarm_handler(_signum, _frame):
                raise ComplexProcessingTimeout(f"exceeded {timeout_seconds}s")
            previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        result = featurizer.featurize_complex(complex_inputs)
        if result is None:
            return {"ok": False, "name": name, "stage": "featurize", "error": "returned None"}
        return {"ok": True, "name": name, "result": result}
    except Exception as error:
        return {
            "ok": False,
            "name": name,
            "stage": "featurize_timeout" if isinstance(error, ComplexProcessingTimeout) else "featurize",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)


@dataclasses.dataclass
class TrainingPipelineConfig:
    dataset: str
    complex_file: str
    data_dir: str
    cache_path: str
    apo_protein_file: str
    holo_protein_file: str
    num_workers: int = 1
    task_timeout_seconds: float = 120.0
    esm_embeddings_path: str = None


class TrainingDataPipeline:
    def __init__(
        self,
        config: TrainingPipelineConfig,
        featurizer_cfg,
    ):
        self.config = config
        self.featurizer = Featurizer.from_config(featurizer_cfg)
        self.parser = ComplexParser(esm_embeddings_path=self.config.esm_embeddings_path)
        self.apo_protein_file = config.apo_protein_file
        self.holo_protein_file = config.holo_protein_file
        self.base_dir = config.data_dir

    def process_all_complexes(self):
        logging.info(
            f"Processing complexes from [{self.config.complex_file}]"
            f"and saving it to [{self.config.cache_path}]"
        )
        os.makedirs(self.config.cache_path, exist_ok=True)
        failures_path = f"{self.config.cache_path}/failed_complexes.jsonl"

        def record_failure(name, stage, error, **extra):
            record = {"name": str(name), "stage": stage, "error": str(error)}
            record.update(extra)
            with open(failures_path, "a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            logging.warning("Skipping %s at %s: %s", name, stage, error)

        complex_names_all = read_strings_from_txt(self.config.complex_file)
        logging.info(f"Loading {len(complex_names_all)} complexes.")

        CHUNK_SIZE = 1000

        # Preserve the manifest on resumptions.  The individual graph files
        # are the source of truth for training, but an empty manifest is
        # misleading when every item is correctly skipped as already cached.
        processed_names = [
            name
            for name in complex_names_all
            if os.path.exists(f"{self.config.cache_path}/heterograph-{name}-0.pt")
        ]

        list_indices = list(range(len(complex_names_all) // CHUNK_SIZE + 1))
        # random.shuffle(list_indices)
        for i in list_indices:
            complex_names = complex_names_all[CHUNK_SIZE * i : CHUNK_SIZE * (i + 1)]

            # DockingDataset loads graphs as ``heterograph-<name>-0.pt``.
            # Skipping completed items makes raw-dataset preprocessing safe to
            # resume after an interruption.
            complex_names = [
                name
                for name in complex_names
                if not os.path.exists(
                    f"{self.config.cache_path}/heterograph-{name}-0.pt"
                )
            ]
            if not complex_names:
                continue

            complex_inputs_shard = []
            for complex_name in complex_names:
                try:
                    parsed = self.parser.parse_complex(self.prepare_input_files(complex_name))
                    if parsed is None:
                        record_failure(complex_name, "parse", "parser returned None")
                        continue
                    complex_inputs_shard.append(parsed)
                except Exception as error:
                    record_failure(complex_name, "parse", f"{type(error).__name__}: {error}",
                                   traceback=traceback.format_exc(limit=8))
            if not complex_inputs_shard:
                continue

            logging.info(f"Num workers={self.config.num_workers}")
            # Stream unordered worker results directly to disk.  This avoids
            # keeping a full 1,000-complex shard in RAM and, crucially,
            # prevents one pathological ligand from delaying cache writes for
            # every completed complex in the shard.
            def save_result(row):
                if not row.get("ok", False):
                    record_failure(row.get("name", "unknown"), row.get("stage", "featurize"),
                                   row.get("error", "unknown error"),
                                   traceback=row.get("traceback", ""))
                    return
                result = row["result"]
                complex_graph = result["complex_graph"]
                ligand = result["ligand"]
                name = result["name"]
                torch.save(complex_graph, f"{self.config.cache_path}/heterograph-{name}-0.pt")
                with open(f"{self.config.cache_path}/rdkit_ligand-{name}-0.pkl", "wb") as f:
                    pickle.dump(ligand[0], f)
                processed_names.append(name)

            try:
                with Parallel(
                    n_jobs=self.config.num_workers,
                    verbose=5,
                    return_as="generator_unordered",
                ) as parallel:
                    results = parallel(
                        delayed(_safe_featurize)(self.featurizer, complex_inputs,
                                                 self.config.task_timeout_seconds)
                        for complex_inputs in complex_inputs_shard
                    )
                    for row in results:
                        save_result(row)
            except Exception as error:
                # A worker crash is rare (normally _safe_featurize catches the
                # error), but retry unfinished items in the parent so a single
                # corrupt worker never loses an entire shard.
                logging.exception("Parallel worker failure; retrying unfinished shard items")
                for complex_inputs in complex_inputs_shard:
                    name = _complex_name(complex_inputs)
                    if not os.path.exists(f"{self.config.cache_path}/heterograph-{name}-0.pt"):
                        save_result(_safe_featurize(
                            self.featurizer, complex_inputs, self.config.task_timeout_seconds
                        ))

        with open(f"{self.config.cache_path}/complex_names.pkl", "wb") as f:
            pickle.dump(processed_names, f)

    def prepare_input_files(self, complex_name):
        if self.config.dataset == "pdbbind":
            complex_dict = {
                "dataset": self.config.dataset,
                "base_dir": self.base_dir,
                "name": complex_name,
                "ligand_description": "filename",
                "ligand_path": f"{self.base_dir}/{complex_name}/{complex_name}_ligand.sdf",
                "apo_rec_path": f"{self.base_dir}/{complex_name}/{complex_name}_{self.apo_protein_file}.pdb",
                "holo_rec_path": f"{self.base_dir}/{complex_name}/{complex_name}_{self.holo_protein_file}.pdb",
            }

        elif self.config.dataset == "plinder":
            complex_dict = {
                "dataset": self.config.dataset,
                "base_dir": self.base_dir,
                "name": complex_name,
                "ligand_description": "filename",
                "apo_rec_path": f"{self.base_dir}/{complex_name}/{self.apo_protein_file}.pdb",
                "holo_rec_path": f"{self.base_dir}/{complex_name}/{self.holo_protein_file}.pdb",
            }
        elif self.config.dataset == "rigid_pocket":
            # The supplied *_protein_256.pdb is already the pocket crop and
            # is used exclusively as fixed conditioning geometry in stage 1.
            complex_dict = {
                "dataset": self.config.dataset,
                "base_dir": self.base_dir,
                "name": complex_name,
                "ligand_description": "filename",
                "ligand_path": f"{self.base_dir}/{complex_name}/{complex_name}_ligand.sdf",
                "apo_rec_path": f"{self.base_dir}/{complex_name}/{complex_name}_{self.apo_protein_file}.pdb",
                "holo_rec_path": None,
            }
        else:
            raise ValueError(f"Unsupported dataset layout: {self.config.dataset}")
        return complex_dict
