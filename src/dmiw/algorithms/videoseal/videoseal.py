import torch
from pydantic import Field
from ..base import BaseAlgorithm, BaseAlgorithmSettings
from typing import Literal
from pathlib import Path
from dmiw.utils import resize_torch_img, overlay_difference
from .cfg import setup_model_from_checkpoint

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent.parent.parent / "models" / "videoseal"
)


class VideoSealSettings(BaseAlgorithmSettings):
    model_folder: Path = Field(
        DEFAULT_MODEL_PATH,
        description="Путь к директории с весами моделей метода Videoseal",
    )
    algorithm_name: Literal["videoseal"] = "videoseal"
    model: Literal["pixelseal", "isp_128"] = Field(
        "isp_128", description="Тип модели алгоритма Videoseal"
    )


class VideoSeal(BaseAlgorithm):
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
        self.wm_strength = wm_strength * 0.2
        self.interpolation_before = interpolation_before
        self.interpolation_after = interpolation_after

        if model == "isp_128":
            model_file = "isp_trained_128.pth"
        elif model == "pixelseal":
            model_file = "pixelseal.pth"
        self.model = setup_model_from_checkpoint(
            str((model_folder / model_file).resolve())
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.model.compile()

    def embed(self, image, key):
        image = image.to(self.device)
        key = key.to(self.device)
        cover = resize_torch_img(
            image,
            [self.model.img_size, self.model.img_size],
            mode=self.interpolation_before,
        )
        cover_norm = cover.unsqueeze(0)
        cover_input = cover_norm
        if self.model.embedder.yuv:
            cover_input = self.model.rgb2yuv(cover_norm)[:, 0:1]
        with torch.no_grad():
            preds_w = self.model.embedder(cover_input, key.unsqueeze(0))

        if self.model.attenuation is not None:
            self.model.attenuation.to(self.device)
            hmaps = self.model.attenuation.heatmaps(cover_norm)
            preds_w = hmaps * preds_w
        marked_img = cover + preds_w
        return overlay_difference(
            image,
            cover,
            marked_img,
            self.wm_strength,
            resize_mode=self.interpolation_after,
        ).cpu()

    def extract(self, image):
        stego = resize_torch_img(
            image,
            [self.model.img_size, self.model.img_size],
            mode=self.interpolation_before,
        )
        stego_norm = stego.unsqueeze(0).to(self.device)
        with torch.no_grad():
            return (
                (self.model.detector(stego_norm).detach().cpu() > 0)
                .squeeze(0)[1:]
                .to(torch.float32)
            )
