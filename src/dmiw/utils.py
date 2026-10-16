import torch
from typing_extensions import NewType, Optional
from torchvision.transforms import Normalize

TorchImg = NewType("TorchImg", torch.Tensor)
TorchImgNormalize = NewType("TorchImgNormalize", torch.Tensor)


def resize_torch_img(
    image: TorchImg,
    size: list[int],
    mode: str = "bilinear",
    align_corners: bool = True,
) -> TorchImg:
    """Resize a torch image tensor to specified dimensions.

    Parameters
    ----------
    image : TorchImg
        Input image tensor in (C, H, W) format
    size : List[int]
        Target size as [height, width]
    mode : str, optional
        Interpolation mode ('nearest', 'bilinear', 'bicubic')
        Default is 'bilinear'
    align_corners : bool, optional
        Flag for align_corners parameter in interpolation
        Default is True

    Returns
    -------
    TorchImg
        Resized image tensor
    """

    if list(image.shape)[1:] == size:
        return image.clone()
    image = image.unsqueeze(0)
    if mode == "lanczos":
        resized_image = torch.nn.functional.interpolate(
            image, size, mode=mode, antialias=True
        ).squeeze(0)
    elif mode not in ["linear", "bilinear", "bicubic"]:
        resized_image = torch.nn.functional.interpolate(
            image, size, mode=mode
        ).squeeze(0)
    else:
        resized_image = torch.nn.functional.interpolate(
            image, size, mode=mode, align_corners=align_corners
        ).squeeze(0)
    return resized_image


def overlay_difference(
    original_image: TorchImg,
    resized_image: TorchImg,
    marked_image: TorchImg,
    factor: float = 1.0,
    resize_mode: str = "bilinear",
) -> TorchImg:
    """Overlay difference between images of one size to image of another size.

    Parameters
    ----------
    original_image : TorchImg
        Base reference image to overlay on
    resized_image : TorchImg
        Resized version of original (should match marked_image size)
    marked_image : TorchImg
        Watermarked or processed image
    factor : float
        Factor to enhance difference

    Returns
    -------
    TorchImg
        Original image with overlay

    Notes
    -----
    - Computes difference between marked and resized images
    - Resizes difference to match original image dimensions
    - Adds difference to original image
    """
    orig_height, orig_width = original_image.shape[1:]
    diff = (marked_image - resized_image) * factor
    min_val = diff.min()
    diff_resized = resize_torch_img(
        (diff - min_val).squeeze(0),
        (orig_height, orig_width),
        mode=resize_mode,
    )
    marked_image = torch.clip(
        original_image + diff_resized + min_val, 0, 1
    ).squeeze(0)
    return marked_image


def normalize_image(
    image: TorchImg, transform: Optional[Normalize] = None
) -> TorchImgNormalize:
    """Normalize image tensor from [0,1] to [-1,1] range and add batch dimension.

    Parameters
    ----------
    image : TorchImg
        Input image in [0,1] range (C, H, W)
    transform : Optional[Normalize], optional
        Custom normalization transform. If None, uses default [-1,1] scaling.

    Returns
    -------
    TorchImgNormalize
        Normalized image in [-1,1] range with batch dimension (1, C, H, W)
    """
    if transform is not None:
        return transform(image).unsqueeze(0)
    return (image * 2 - 1).unsqueeze(0)


def denormalize_image(
    image: TorchImgNormalize, transform: Optional[Normalize] = None
) -> TorchImg:
    """Denormalize image tensor from [-1,1] to [0,1] range and remove batch dimension.

    Parameters
    ----------
    image : TorchImgNormalize
        Input image in [-1,1] range (1, C, H, W)
    transform : Optional[Normalize], optional
        Custom denormalization transform. If None, uses default scaling.

    Returns
    -------
    TorchImg
        Denormalized image in [0,1] range (C, H, W)
    """
    if transform is not None:
        return transform(image).squeeze(0)
    return ((image + 1) / 2).squeeze(0)
