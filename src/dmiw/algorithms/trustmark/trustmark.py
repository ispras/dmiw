import torch
from pydantic import Field
from ..base import BaseAlgorithm, BaseAlgorithmSettings
from typing import Literal
from pathlib import Path
from omegaconf import OmegaConf
import importlib
from dmiw.utils import (
    resize_torch_img,
    overlay_difference,
    normalize_image,
    denormalize_image,
)

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent.parent.parent / "models" / "trustmark"
)


class TrustMarkSettings(BaseAlgorithmSettings):
    model_folder: Path = Field(
        DEFAULT_MODEL_PATH,
        description="Путь к директории с весами моделей метода Trustmark",
    )
    algorithm_name: Literal["trustmark"] = "trustmark"
    model: Literal["P", "Q"] = Field(
        "P", description="Тип модели алгоритма TrustMark"
    )


class TrustMark(BaseAlgorithm):
    def __init__(
        self,
        model_folder: Path,
        model: Literal["P", "Q"],
        wm_strength: float,
        device: str,
        interpolation_before: str,
        interpolation_after: str,
    ):
        super().__init__()
        self.model_type = model
        self.device = device
        self.wm_strength = wm_strength * 0.75
        self.interpolation_before = interpolation_before
        self.interpolation_after = interpolation_after
        if self.model_type == "P":
            self.wm_strength = wm_strength * 1.25

        if self.model_type == "P":
            self.model_resolution_enc = 256
            self.model_resolution_dec = 224
            self.aspect_ratio_lim = 0  # always force to centre square crop
        else:
            self.model_resolution_enc = 256
            self.model_resolution_dec = 245
        self.model_resolution_remove = 256
        self.model_resolution_max_detect = 1600

        locations = {
            "config": model_folder / f"trustmark_{self.model_type}.yaml",
            "encoder": model_folder / f"encoder_{self.model_type}.ckpt",
            "decoder": model_folder / f"decoder_{self.model_type}.ckpt",
        }

        self.decoder = self.load_model(
            locations["config"],
            locations["decoder"],
            self.device,
            part="decoder",
        )
        self.encoder = self.load_model(
            locations["config"],
            locations["encoder"],
            self.device,
            part="encoder",
        )

    def load_model(self, config_path, weight_path, device, part):
        if not config_path.exists():
            raise FileNotFoundError(f"Файл не найден: {config_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"Файл не найден: {weight_path}")

        config = OmegaConf.load(config_path).model
        if part == "encoder":
            # replace all other components with identity
            config.params.secret_decoder_config.target = (
                "trustmark.model.Identity"
            )
            config.params.discriminator_config.target = (
                "trustmark.model.Identity"
            )
            config.params.loss_config.target = "trustmark.model.Identity"
            config.params.noise_config.target = "trustmark.model.Identity"
        elif part == "decoder":
            # replace all other components with identity
            config.params.secret_encoder_config.target = (
                "trustmark.model.Identity"
            )
            config.params.discriminator_config.target = (
                "trustmark.model.Identity"
            )
            config.params.loss_config.target = "trustmark.model.Identity"
            config.params.noise_config.target = "trustmark.model.Identity"

        model = self.instantiate_from_config(config)
        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))

        if "global_step" in state_dict:
            print(
                f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}'
            )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        misses, ignores = model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        return model

    def embed(self, image, key):
        image = image.to(self.device)
        key = key.to(self.device)
        cover = resize_torch_img(
            image,
            [self.model_resolution_enc, self.model_resolution_enc],
            mode=self.interpolation_before,
        )
        cover_norm = normalize_image(cover)
        with torch.no_grad():
            stego, _ = self.encoder(cover_norm, key)
        denorm_stego = denormalize_image(stego.detach())
        return overlay_difference(
            image,
            cover,
            denorm_stego,
            self.wm_strength,
            resize_mode=self.interpolation_after,
        ).cpu()

    def extract(self, image):
        image = image.to(self.device)
        stego = resize_torch_img(
            image, [self.model_resolution_dec, self.model_resolution_dec]
        )
        stego_norm = normalize_image(stego)
        with torch.no_grad():
            return (self.decoder.decoder(stego_norm).detach().cpu() > 0).to(
                torch.float32
            )

    def instantiate_from_config(self, config):
        if not "target" in config:
            if config == "__is_first_stage__":
                return None
            elif config == "__is_unconditional__":
                return None
            raise KeyError("Expected key `target` to instantiate.")
        return self.get_obj_from_str(config["target"])(
            **config.get("params", dict())
        )

    @staticmethod
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        module = module.replace("trustmark", __package__)
        return getattr(importlib.import_module(module), cls)
