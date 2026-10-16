import omegaconf
import torch
import os
import torch.distributed as dist
import torchvision.transforms as transforms
from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from .models import Videoseal, build_embedder, build_extractor, build_baseline
from .augmentation.augmenter import get_dummy_augmenter
from .jnd import JND


@dataclass
class SubModelConfig:
    """Configuration for a sub-model."""

    model: str
    params: DictConfig


@dataclass
class VideosealConfig:
    """Configuration for a Video Seal model."""

    args: DictConfig
    embedder: SubModelConfig
    extractor: SubModelConfig


def resolve_config_path(cfg_path):
    # Resolve the config path in the following order:
    # 1. Search from the working directory
    # 2. Search from the package directory (for pip-installed packages)
    if not Path(cfg_path).is_file():
        # Use parents[1] to resolve relative to the videoseal package root
        cfg_path = Path(__file__).parents[1].joinpath(cfg_path)

    return cfg_path


def setup_model(config: VideosealConfig, ckpt_path: Path) -> Videoseal:
    """
    Set up a Video Seal model from a configuration and checkpoint file.

    Args:
    config (VideosealConfig): Model configuration.
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    Videoseal: Loaded model.
    """
    args = config.args

    # prepare some args for backward compatibility
    if "img_size_proc" in args:
        args.img_size = args.img_size_proc
    else:
        args.img_size = args.img_size_extractor

    if "hidden_size_multiplier" in args:
        args.hidden_size_multiplier = args.hidden_size_multiplier
    else:
        args.hidden_size_multiplier = 2

    # Handle backward compatibility for videowam_* parameter names (from 1210-oss branch)
    # If old parameter names exist (videowam_chunk_size, videowam_step_size),
    # use them to set the new names (videoseal_chunk_size, videoseal_step_size)
    if "videowam_chunk_size" in args and "videoseal_chunk_size" not in args:
        args.videoseal_chunk_size = args.videowam_chunk_size
    if "videowam_step_size" in args and "videoseal_step_size" not in args:
        args.videoseal_step_size = args.videowam_step_size

    # Build models
    embedder = build_embedder(
        config.embedder.model,
        config.embedder.params,
        args.nbits,
        args.hidden_size_multiplier,
    )
    extractor = build_extractor(
        config.extractor.model,
        config.extractor.params,
        args.img_size,
        args.nbits,
    )
    augmenter = get_dummy_augmenter()
    # Build attenuation
    if args.attenuation.lower().startswith("jnd"):
        ckpt_path = Path(ckpt_path)
        attenuation_cfg_path = ckpt_path.with_name(
            ckpt_path.stem + "_attenuation.yaml"
        )
        attenuation_cfg = omegaconf.OmegaConf.load(attenuation_cfg_path)
        attenuation = JND(**attenuation_cfg[args.attenuation])
    else:
        attenuation = None

    # Build the complete model
    wam = Videoseal(
        embedder,
        extractor,
        augmenter=augmenter,
        attenuation=attenuation,
        scaling_w=args.scaling_w,
        scaling_i=args.scaling_i,
        img_size=args.img_size,
        chunk_size=args.videoseal_chunk_size,
        step_size=args.videoseal_step_size,
    )

    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        msg = wam.load_state_dict(checkpoint["model"], strict=False)
        # print(f"Model loaded successfully from {ckpt_path} with message: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    return wam


def get_config_from_checkpoint(ckpt_path: Path) -> VideosealConfig:
    """
    Load configuration from a checkpoint file.

    Args:
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    VideosealConfig: Loaded configuration.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    args = OmegaConf.create(args)

    if not isinstance(args, DictConfig):
        raise Exception("Expected logfile to contain params dictionary.")
    ckpt_path = Path(ckpt_path)
    # Load sub-model configurations
    embedder_config = ckpt_path.with_name(ckpt_path.stem + "_embedder.yaml")
    embedder_cfg_path = resolve_config_path(embedder_config)
    embedder_cfg = OmegaConf.load(embedder_cfg_path)
    extractor_config = ckpt_path.with_name(ckpt_path.stem + "_extractor.yaml")
    extractor_cfg_path = resolve_config_path(extractor_config)
    extractor_cfg = OmegaConf.load(extractor_cfg_path)

    # Create sub-model configurations
    embedder_model = args.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[embedder_model]
    extractor_model = args.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[extractor_model]

    return VideosealConfig(
        args=args,
        embedder=SubModelConfig(model=embedder_model, params=embedder_params),
        extractor=SubModelConfig(
            model=extractor_model, params=extractor_params
        ),
    )


def setup_model_from_checkpoint(ckpt_path: str) -> Videoseal:
    """
    # Example usage
    ckpt_path = '/path/to/videoseal/checkpoint.pth'
    wam = setup_model_from_checkpoint(ckpt_path)

    or
    ckpt_path = 'baseline/wam'
    wam = setup_model_from_checkpoint(ckpt_path)
    """
    # load baselines. Should be in the format of "baseline/{method}"
    if "baseline" in ckpt_path:
        method = ckpt_path.split("/")[-1]
        return build_baseline(method)
    config = get_config_from_checkpoint(ckpt_path)
    return setup_model(config, ckpt_path)
