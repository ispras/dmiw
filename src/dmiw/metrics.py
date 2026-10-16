from .utils import TorchImg, resize_torch_img, normalize_image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import torch
from lpips import LPIPS
import warnings


def psnr(
    img1: TorchImg,
    img2: TorchImg,
) -> float:
    if torch.equal(img1, img2):
        return float("inf")
    img2 = resize_torch_img(img2, list(img1.shape)[1:])
    return float(psnr_func(img1.numpy(), img2.numpy(), data_range=1))


def ssim(
    img1: TorchImg,
    img2: TorchImg,
) -> float:
    img2 = resize_torch_img(img2, list(img1.shape)[1:])
    if len(img1.shape) == 2:
        return float(ssim_func(img1.numpy(), img2.numpy(), data_range=1))
    res = ssim_func(img1.numpy(), img2.numpy(), data_range=1, channel_axis=0)
    return float(res)


def lpips(
    img1: TorchImg,
    img2: TorchImg,
    net: str = "alex",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_fn = LPIPS(net=net, verbose=False).to(device)
    img2 = resize_torch_img(img2, list(img1.shape)[1:])
    with torch.no_grad():
        return float(
            loss_fn(
                normalize_image(img1).to(device),
                normalize_image(img2).to(device),
            )
        )
