from ifmdock.models.pl_modules.docking import IFMDockModule
from ifmdock.models.pl_modules.filtering import FilteringModule
from ifmdock.models.pl_modules.relaxation import RelaxFlowModule


TASK_TO_MODULES = {
    "docking": IFMDockModule,
    "filtering": FilteringModule,
    "relaxation": RelaxFlowModule,
}


def setup_model(cfg, task: str = "docking"):
    model_cls = TASK_TO_MODULES.get(task, None)
    if model_cls is None:
        raise ValueError(
            f"Task of type={task} not supported. Supported tasks are {list(TASK_TO_MODULES.keys())}"
        )

    if task == "docking":
        return IFMDockModule(
            model_cfg=cfg.model,
            sigma_cfg=cfg.sigma,
            training_cfg=cfg.training,
            sampler_cfg=cfg.sampler,
            loss_cfg=cfg.loss,
            physical_loss_cfg=cfg.get("physical_loss", None),
        )
    else:
        raise NotImplementedError(
            "Config based pipeline is WIP for confidence and relaxation"
        )
