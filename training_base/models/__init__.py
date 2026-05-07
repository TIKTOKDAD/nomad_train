from dataclasses import dataclass
from typing import Any, Dict

from training_base.registry import model_registry, module_registry, noise_scheduler_registry


@dataclass
class ModelBuild:
    model: Any
    extras: Dict[str, Any]


@model_registry.register("gnm")
def build_gnm(config) -> ModelBuild:
    from training_base.models.gnm import GNM

    data = config["data"]
    model_config = config["model"]
    encoder_config = dict(model_config)
    if "encoder" in model_config:
        encoder_config.update(model_config["encoder"])
        encoder_name = encoder_config.get("name", "gnm_encoder")
    else:
        encoder_name = "gnm_encoder"
    encoder = module_registry.build(encoder_name, encoder_config, data)
    head = module_registry.build(
        model_config.get("head", {}).get("name", "waypoint_head"),
        {
            "input_dim": encoder.output_dim,
            "len_traj_pred": data["len_traj_pred"],
            "num_action_params": 4 if data["learn_angle"] else 2,
            "learn_angle": data["learn_angle"],
        },
    )
    model = GNM(
        context_size=data["context_size"],
        len_traj_pred=data["len_traj_pred"],
        learn_angle=data["learn_angle"],
        encoder=encoder,
        head=head,
    )
    return ModelBuild(model=model, extras={})


@model_registry.register("vint")
def build_vint(config) -> ModelBuild:
    from training_base.models.vint import ViNT

    data = config["data"]
    model_config = config["model"]
    encoder_config = dict(model_config)
    if "encoder" in model_config:
        encoder_config.update(model_config["encoder"])
        encoder_name = encoder_config.get("name", "vint_encoder")
    else:
        encoder_name = "vint_encoder"
    encoder = module_registry.build(encoder_name, encoder_config, data)
    head = module_registry.build(
        model_config.get("head", {}).get("name", "waypoint_head"),
        {
            "input_dim": encoder.output_dim,
            "len_traj_pred": data["len_traj_pred"],
            "num_action_params": 4 if data["learn_angle"] else 2,
            "learn_angle": data["learn_angle"],
        },
    )
    model = ViNT(
        context_size=data["context_size"],
        len_traj_pred=data["len_traj_pred"],
        learn_angle=data["learn_angle"],
        encoder=encoder,
        head=head,
    )
    return ModelBuild(model=model, extras={})


@model_registry.register("nomad")
def build_nomad(config) -> ModelBuild:
    from training_base.models.nomad import NoMaD

    model_config = config["model"]
    vision_config = model_config["vision_encoder"]
    vision_encoder = module_registry.build(vision_config["name"], vision_config, config["data"])
    diffusion_config = dict(model_config["diffusion"])
    diffusion_config.setdefault("global_cond_dim", model_config["vision_encoder"]["encoding_size"])
    diffusion_model = module_registry.build(model_config["diffusion"]["name"], diffusion_config)
    distance_predictor = module_registry.build(
        model_config["distance_predictor"]["name"],
        model_config["distance_predictor"]["embedding_dim"],
    )
    model = NoMaD(
        vision_encoder=vision_encoder,
        diffusion_model=diffusion_model,
        distance_predictor=distance_predictor,
    )
    scheduler = noise_scheduler_registry.build(model_config["diffusion_scheduler"]["name"], model_config["diffusion_scheduler"])
    return ModelBuild(model=model, extras={"noise_scheduler": scheduler})


def build_model(config) -> ModelBuild:
    return model_registry.build(config["model"]["name"], config)
