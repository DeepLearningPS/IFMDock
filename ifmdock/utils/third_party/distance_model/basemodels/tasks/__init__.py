"""isort:skip_file"""

import argparse
import importlib
import os

from .task import UnicoreTask
TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def setup_task(args, **kwargs):
    return TASK_REGISTRY[args.task].setup_task(args, **kwargs)


def register_task(name):
    """
    New tasks can be added to unicore with the
    :func:`~Distance_model.basemodels.tasks.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(UnicoreTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~Distance_model.basemodels.tasks.UnicoreTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, UnicoreTask):
            raise ValueError(
                "Task ({}: {}) must extend UnicoreTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        task_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("Distance_model.basemodels.tasks." + task_name)
        if task_name in TASK_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_task = parser.add_argument_group("Task name")
            group_task.add_argument('--task', metavar=task_name,
                                    help='Enable this task with: ``--task=' + task_name + '``')
            group_args = parser.add_argument_group("Additional command-line arguments")
            TASK_REGISTRY[task_name].add_args(group_args)
            globals()[task_name + "_parser"] = parser
