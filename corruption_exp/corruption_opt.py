"""Image corruption operations used by the corruption experiments.

The parameter choices follow the public Moderate-DS corruption scripts:

- Gaussian noise: mean=0.2, standard deviation=1.0 in [0, 1] space.
- Partial occlusion:
    CIFAR-sized images: rectangle (5, 5, 20, 20)
    Tiny-ImageNet-sized images: rectangle (5, 5, 45, 45)
- Resolution degradation:
    32x32 -> 8x8 -> 32x32
    64x64 -> 16x16 -> 64x64
- Fog: atmospheric light A=0.5 and beta=0.4.
- Motion blur: degree=15 and angle=45 degrees.

Floating-point outputs are clipped to [0, 1] before conversion to uint8.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageDraw


ImageInput: TypeAlias = Image.Image | np.ndarray

GAUSSIAN_NOISE = 0
PARTIAL_OCCLUSION = 1
RESOLUTION_DEGRADATION = 2
FOG = 3
MOTION_BLUR = 4

NUM_CORRUPTION_TYPES = 5

CORRUPTION_ID_TO_NAME: dict[int, str] = {
    GAUSSIAN_NOISE: "gaussian_noise",
    PARTIAL_OCCLUSION: "partial_occlusion",
    RESOLUTION_DEGRADATION: "resolution_degradation",
    FOG: "fog",
    MOTION_BLUR: "motion_blur",
}
CORRUPTION_NAME_TO_ID: dict[str, int] = {
    name: type_id for type_id, name in CORRUPTION_ID_TO_NAME.items()
}


def _to_rgb_uint8_array(image: ImageInput) -> np.ndarray:
    """Convert a PIL image or NumPy array to an HWC RGB uint8 array."""
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(
            f"image must be PIL.Image.Image or numpy.ndarray, got {type(image)!r}"
        )

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(
            "Expected an HWC image with 1, 3, or 4 channels, "
            f"got shape={array.shape}."
        )
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)

    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def _to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.ascontiguousarray(array.astype(np.uint8)),
        mode="RGB",
    )


def gaussian_noise(
    image: ImageInput,
    rng: np.random.Generator | None = None,
    *,
    mean: float = 0.2,
    std: float = 1.0,
) -> Image.Image:
    """Add Gaussian noise in normalized [0, 1] RGB space."""
    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}.")
    generator = rng if rng is not None else np.random.default_rng()

    array = _to_rgb_uint8_array(image).astype(np.float32) / 255.0
    noise = generator.normal(
        loc=mean,
        scale=std,
        size=array.shape,
    ).astype(np.float32)
    corrupted = np.clip(array + noise, 0.0, 1.0)
    return _to_pil(np.rint(corrupted * 255.0))


def partial_occlusion(image: ImageInput) -> Image.Image:
    """Draw the fixed black rectangle used in the Moderate-DS scripts."""
    pil_image = _to_pil(_to_rgb_uint8_array(image))
    width, height = pil_image.size

    if width == 32 and height == 32:
        box = (5, 5, 20, 20)
    elif width == 64 and height == 64:
        box = (5, 5, 45, 45)
    else:
        left = max(0, int(round(width * 5 / 64)))
        top = max(0, int(round(height * 5 / 64)))
        right = min(width - 1, int(round(width * 45 / 64)))
        bottom = min(height - 1, int(round(height * 45 / 64)))
        box = (left, top, right, bottom)

    output = pil_image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle(box, fill=(0, 0, 0))
    return output


def resolution_degradation(image: ImageInput) -> Image.Image:
    """Downsample to one quarter of each spatial dimension, then upsample."""
    pil_image = _to_pil(_to_rgb_uint8_array(image))
    width, height = pil_image.size

    if width == 32 and height == 32:
        low_size = (8, 8)
    elif width == 64 and height == 64:
        low_size = (16, 16)
    else:
        low_size = (max(1, width // 4), max(1, height // 4))

    resampling = getattr(Image, "Resampling", Image)
    low_res = pil_image.resize(low_size, resample=resampling.BILINEAR)
    return low_res.resize((width, height), resample=resampling.BILINEAR)


def fog(
    image: ImageInput,
    *,
    atmospheric_light: float = 0.5,
    beta: float = 0.4,
) -> Image.Image:
    """Apply the Moderate-DS radial fog model with safe uint8 conversion."""
    if not 0.0 <= atmospheric_light <= 1.0:
        raise ValueError(
            "atmospheric_light must be in [0, 1], got "
            f"{atmospheric_light}."
        )
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}.")

    array = _to_rgb_uint8_array(image).astype(np.float32) / 255.0
    height, width, _ = array.shape

    y, x = np.ogrid[:height, :width]
    center_y = height // 2
    center_x = width // 2
    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    size = np.sqrt(max(height, width))
    depth = -0.04 * distance + size
    transmission = np.exp(-beta * depth).astype(np.float32)[..., None]

    corrupted = (
        array * transmission
        + atmospheric_light * (1.0 - transmission)
    )
    corrupted = np.clip(corrupted, 0.0, 1.0)
    return _to_pil(np.rint(corrupted * 255.0))


def motion_blur(
    image: ImageInput,
    *,
    degree: int = 15,
    angle: float = 45.0,
) -> Image.Image:
    """Apply a normalized linear motion-blur kernel."""
    if degree <= 0:
        raise ValueError(f"degree must be positive, got {degree}.")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "motion_blur requires opencv-python. Install it with "
            "`pip install opencv-python`."
        ) from exc

    array = _to_rgb_uint8_array(image)

    kernel = np.diag(np.ones(degree, dtype=np.float32))
    center = ((degree - 1) / 2.0, (degree - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (degree, degree))

    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        raise RuntimeError("Generated motion-blur kernel has non-positive sum.")
    kernel /= kernel_sum

    blurred = cv2.filter2D(
        array,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return _to_pil(np.clip(blurred, 0, 255).astype(np.uint8))


def apply_corruption(
    image: ImageInput,
    corruption_type: int,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Apply one corruption selected by its integer type ID.

    Only Gaussian noise consumes ``rng``. Keeping it in the shared interface lets
    data-generation code dispatch every operation uniformly and reproducibly.
    """
    if corruption_type == GAUSSIAN_NOISE:
        return gaussian_noise(image, rng=rng)
    if corruption_type == PARTIAL_OCCLUSION:
        return partial_occlusion(image)
    if corruption_type == RESOLUTION_DEGRADATION:
        return resolution_degradation(image)
    if corruption_type == FOG:
        return fog(image)
    if corruption_type == MOTION_BLUR:
        return motion_blur(image)

    raise ValueError(
        f"Unknown corruption_type={corruption_type}. "
        f"Valid IDs are {sorted(CORRUPTION_ID_TO_NAME)}."
    )


CORRUPTION_FUNCTIONS: dict[int, Callable[..., Image.Image]] = {
    GAUSSIAN_NOISE: gaussian_noise,
    PARTIAL_OCCLUSION: partial_occlusion,
    RESOLUTION_DEGRADATION: resolution_degradation,
    FOG: fog,
    MOTION_BLUR: motion_blur,
}


__all__ = [
    "GAUSSIAN_NOISE",
    "PARTIAL_OCCLUSION",
    "RESOLUTION_DEGRADATION",
    "FOG",
    "MOTION_BLUR",
    "NUM_CORRUPTION_TYPES",
    "CORRUPTION_ID_TO_NAME",
    "CORRUPTION_NAME_TO_ID",
    "CORRUPTION_FUNCTIONS",
    "gaussian_noise",
    "partial_occlusion",
    "resolution_degradation",
    "fog",
    "motion_blur",
    "apply_corruption",
]
