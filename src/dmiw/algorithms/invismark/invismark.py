import torch
import sys
from pydantic import Field
from ..base import BaseAlgorithm, BaseAlgorithmSettings
from typing import Literal
from pathlib import Path
from dmiw.utils import (
    resize_torch_img,
    overlay_difference,
    normalize_image,
    denormalize_image,
)
from .model import Encoder, Extractor

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent.parent.parent / "models" / "invismark"
)


class InvisMarkSettings(BaseAlgorithmSettings):
    model_folder: Path = Field(
        DEFAULT_MODEL_PATH,
        description="Путь к директории с весами моделей метода Invismark",
    )
    algorithm_name: Literal["invismark"] = "invismark"
    model: Literal["invismark", "isp"] = Field(
        "isp", description="Тип модели алгоритма InvisMark"
    )


class InvisMark(BaseAlgorithm):
    def __init__(
        self,
        model_folder: Path,
        model: str,
        wm_strength: float,
        device: str,
        interpolation_before: str,
        interpolation_after: str,
    ):
        super().__init__()
        self.device = device
        self.wm_strength = wm_strength
        self.interpolation_before = interpolation_before
        self.interpolation_after = interpolation_after
        if model == "invismark":
            model_file = "invismark.ckpt"
        elif model == "isp":
            model_file = "isp_trained.ckpt"

        sys.path.append(str(Path(__file__).parent))
        state_dict = torch.load(
            model_folder / model_file,
            map_location=self.device,
            weights_only=False,
        )
        config = state_dict["config"]
        self.config = config
        self.encoder = Encoder(config).to(device)
        self.decoder = Extractor(config).to(device)
        self.load_model(state_dict)

    def load_model(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder_state_dict"])
        self.encoder.eval()
        self.decoder.load_state_dict(state_dict["decoder_state_dict"])
        self.decoder.eval()

    def embed(self, image, key):
        resized_img = resize_torch_img(
            image, self.config.image_shape, mode=self.interpolation_before
        )
        trans_img = normalize_image(resized_img).to(self.device)
        wm = key.unsqueeze(0)
        with torch.no_grad():
            output = self.encoder(
                trans_img, wm.type(torch.float32).to(self.device)
            )
        res_denorm = denormalize_image(output).cpu()
        res = overlay_difference(
            image,
            resized_img,
            res_denorm,
            self.wm_strength,
            resize_mode=self.interpolation_after,
        )
        return res

    def extract(self, image):
        resized_img = resize_torch_img(
            image, self.config.image_shape, mode=self.interpolation_before
        )
        trans_img = normalize_image(resized_img).to(self.device)
        with torch.no_grad():
            dec = self.decoder(trans_img)
        extracted = torch.round(dec).type(torch.float64)
        return extracted.cpu()
