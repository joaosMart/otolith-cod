"""
Canonical image preprocessing for every stage of the pipeline.

This module is the single definition of how an otolith image becomes a tensor.
Feature extraction, LoRA fine-tuning, full fine-tuning and attention visualisation
all build their transforms from here, so a preprocessing change cannot silently
apply to one stage and not another.

Two things are deliberate:

1. `PreprocessConfig` carries every knob that affects pixel values, and `slug()`
   turns it into a filename fragment. Every cached artifact is named with that
   slug, so two runs with different preprocessing cannot overwrite each other.

2. Train and eval transforms are derived from the same processor-supplied
   resolution and normalisation statistics. Only the augmentation differs.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import PIL.Image as Image
from torchvision import transforms


# Applied once, not twice. The published CLAHE ablation (frozen SigLIP2 accuracy
# 0.566 -> 0.573) was measured with a single pass, and every cached embedding the
# paper referenced was named without the `_repeat` marker.
DEFAULT_CLIP_LIMIT = 3.0
DEFAULT_TILE_SIZE = 25
DEFAULT_REPEAT = False


#: How a non-square otolith crop is fitted to the encoder's square input.
#:
#: This matters more here than it usually would. The crops are wide and short:
#: median 787x333 px, aspect ratio about 2.4:1, and for three quarters of them
#: the short side is already below the encoder's input resolution. The choice
#: therefore trades geometric fidelity against usable pixel area.
#:
#:   pad      Resize the long side to fit and pad the short side. Aspect ratio
#:            and increment spacing are preserved exactly and no part of the
#:            otolith is discarded, at the cost of spending roughly 58% of the
#:            canvas on padding.
#:   squash   Resize both axes to the target independently. Uses the whole
#:            canvas but compresses the horizontal axis about 2.4x relative to
#:            the vertical, so increment spacing becomes direction-dependent.
#:   crop     Resize the short side to the target and centre-crop the long axis.
#:            Undistorted, but discards the otolith's proximal and distal ends,
#:            which is where the outermost and most age-diagnostic increments sit.
#:
#: The previous pipeline used `squash` when training and `crop` when extracting
#: embeddings, which meant the encoder saw differently-shaped otoliths in the two
#: stages. `pad` is the default because it is the only option that neither
#: distorts increment geometry nor throws away the margins.
RESIZE_MODES = ("pad", "squash", "crop")
DEFAULT_RESIZE_MODE = "pad"


@dataclass(frozen=True)
class PreprocessConfig:
    """Everything that affects the pixels handed to an encoder.

    Frozen so it can be hashed and safely shared between a dataset and the
    transform built from it.
    """

    apply_clahe: bool = True
    repeat_clahe: bool = DEFAULT_REPEAT
    clip_limit: float = DEFAULT_CLIP_LIMIT
    tile_size: int = DEFAULT_TILE_SIZE
    resize_mode: str = DEFAULT_RESIZE_MODE

    def __post_init__(self):
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(
                f"resize_mode must be one of {RESIZE_MODES}, got '{self.resize_mode}'"
            )

    def slug(self) -> str:
        """Filename fragment identifying this configuration.

        Every setting that changes a pixel appears here, so two runs with
        different preprocessing cannot land on the same cache file. That silent
        collision is what previously made CLAHE and non-CLAHE LoRA embeddings
        overwrite each other.
        """
        parts = []
        if self.apply_clahe:
            parts.append("clahe")
            if self.repeat_clahe:
                parts.append("repeat")
            if self.clip_limit != DEFAULT_CLIP_LIMIT:
                parts.append(f"cl{self.clip_limit:g}")
            if self.tile_size != DEFAULT_TILE_SIZE:
                parts.append(f"ts{self.tile_size}")
        if self.resize_mode != DEFAULT_RESIZE_MODE:
            parts.append(self.resize_mode)
        return "_".join(parts)

    def describe(self) -> str:
        """One-line human-readable summary, for logs and saved run metadata."""
        resize = {
            "pad": "aspect-preserving pad to square",
            "squash": "resized to square, aspect ratio not preserved",
            "crop": "short side resized then centre-cropped",
        }[self.resize_mode]
        if not self.apply_clahe:
            return f"no CLAHE, {resize}"
        passes = "twice" if self.repeat_clahe else "once"
        return (
            f"CLAHE applied {passes}, clip limit {self.clip_limit:g}, "
            f"adaptive tile grid at {self.tile_size}px, {resize}"
        )


def clahe_enhancement(
    image: Image.Image,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_size: int = DEFAULT_TILE_SIZE,
    repeat_clahe: bool = DEFAULT_REPEAT,
) -> Image.Image:
    """Contrast-limited adaptive histogram equalisation on a PIL image.

    The tile grid is derived from the image dimensions rather than fixed, so the
    equalisation neighbourhood covers a constant physical area regardless of how
    large the crop is. Returns a 3-channel image because the encoders expect RGB.
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape[:2]
    grid_x = max(1, w // tile_size)
    grid_y = max(1, h // tile_size)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_x, grid_y))
    enhanced = clahe.apply(gray)
    if repeat_clahe:
        enhanced = clahe.apply(enhanced)

    return Image.fromarray(np.stack([enhanced] * 3, axis=-1))


def apply_config(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    """Apply a PreprocessConfig to a PIL image."""
    if not config.apply_clahe:
        return image.convert("RGB")
    return clahe_enhancement(
        image,
        clip_limit=config.clip_limit,
        tile_size=config.tile_size,
        repeat_clahe=config.repeat_clahe,
    )


def fit_to_square(image: Image.Image, size: int, mode: str) -> Image.Image:
    """Fit a non-square image to a square canvas under the chosen policy.

    Padding uses the image's own median intensity rather than black, so the
    border does not introduce a hard high-contrast edge that CLAHE has already
    amplified and that the encoder would read as structure.
    """
    if mode == "squash":
        return image.resize((size, size), Image.BICUBIC)

    w, h = image.size

    if mode == "crop":
        scale = size / min(w, h)
        resized = image.resize(
            (max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC
        )
        rw, rh = resized.size
        left, top = (rw - size) // 2, (rh - size) // 2
        return resized.crop((left, top, left + size, top + size))

    # mode == "pad"
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BICUBIC)

    fill = int(np.median(np.asarray(resized.convert("L"))))
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def resolve_image_size(processor, fallback: int = 384) -> int:
    """Read the target square resolution out of a HuggingFace image processor.

    Processors disagree on which key they use, so all three spellings are tried
    before falling back.
    """
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        for key in ("height", "shortest_edge", "width"):
            if key in size:
                return int(size[key])
    elif isinstance(size, int):
        return int(size)
    return fallback


def _normalization(processor, image_mean, image_std):
    mean = image_mean if image_mean is not None else getattr(processor, "image_mean", None)
    std = image_std if image_std is not None else getattr(processor, "image_std", None)
    if mean is None or std is None:
        raise ValueError(
            "Could not determine normalisation statistics. Pass image_mean and "
            "image_std explicitly for encoders whose processor does not expose them."
        )
    return transforms.Normalize(mean=mean, std=std)


def build_eval_transform(
    processor,
    config: PreprocessConfig,
    image_size: Optional[int] = None,
    image_mean: Optional[Sequence[float]] = None,
    image_std: Optional[Sequence[float]] = None,
):
    """Deterministic transform used for feature extraction and validation.

    Built explicitly rather than by calling the processor, so that it shares its
    resize and normalisation with the training transform below. Calling the
    processor for eval and hand-building the transform for training is how the
    two paths drift apart.
    """
    size = image_size or resolve_image_size(processor)
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_config(img, config)),
        transforms.Lambda(lambda img: fit_to_square(img, size, config.resize_mode)),
        transforms.ToTensor(),
        _normalization(processor, image_mean, image_std),
    ])


def build_train_transform(
    processor,
    config: PreprocessConfig,
    image_size: Optional[int] = None,
    image_mean: Optional[Sequence[float]] = None,
    image_std: Optional[Sequence[float]] = None,
    rotation_degrees: float = 15.0,
    scale_range: tuple = (0.85, 1.15),
    hflip_prob: float = 0.5,
    blur_prob: float = 0.5,
):
    """Training transform: the eval transform plus geometric and blur augmentation.

    The horizontal flip is safe here because left and right otoliths from the same
    fish are mirror images of one another, so a flipped section is a plausible
    specimen rather than an impossible one. Rotation and scale are kept mild
    because growth increment spacing is the signal and aggressive scaling distorts
    it. The blur discourages the encoder from keying on sensor noise.
    """
    size = image_size or resolve_image_size(processor)
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_config(img, config)),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        # Rotation happens before the square fit, on the original geometry, and
        # fills exposed corners with mid-grey rather than black so the encoder is
        # not handed a hard synthetic edge.
        transforms.RandomAffine(degrees=rotation_degrees, scale=scale_range, fill=128),
        transforms.Lambda(lambda img: fit_to_square(img, size, config.resize_mode)),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=blur_prob
        ),
        transforms.ToTensor(),
        _normalization(processor, image_mean, image_std),
    ])
