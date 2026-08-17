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
import torch
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

#: Training-time augmentation set.
#:
#:   base     Horizontal flip, mild rotation and scale, occasional blur. What
#:            every result in the paper up to Section 3.11 was trained with.
#:   strong   `base` plus crop-box jitter, an elastic deformation applied only
#:            across the reading axis, CLAHE with parameters drawn per image,
#:            a smooth illumination field, and random placement within the
#:            padded canvas. Motivated by the learning curves: the plateau
#:            says more otoliths of the same kind add nothing, which is an
#:            argument for varying the ones already held, not for collecting
#:            more.
AUGMENT_MODES = ("base", "strong")
DEFAULT_AUGMENT = "base"


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
    augment: str = DEFAULT_AUGMENT

    def __post_init__(self):
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(
                f"resize_mode must be one of {RESIZE_MODES}, got '{self.resize_mode}'"
            )
        if self.augment not in AUGMENT_MODES:
            raise ValueError(
                f"augment must be one of {AUGMENT_MODES}, got '{self.augment}'"
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
        # In the slug because it changes training pixels, so an augmented run
        # must not be able to overwrite the cache of an unaugmented one.
        if self.augment != DEFAULT_AUGMENT:
            parts.append(f"aug{self.augment}")
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


def fit_to_square(image: Image.Image, size: int, mode: str,
                  rng=None) -> Image.Image:
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
    # Centred unless a generator is supplied. Placing the section randomly in
    # the canvas is a real augmentation rather than a no-op because SigLIP2 and
    # CLIP carry learned position embeddings, so translation invariance is not
    # free for them; with 58% of the canvas empty there is room to use.
    if rng is None:
        offset = ((size - new_w) // 2, (size - new_h) // 2)
    else:
        offset = (int(rng.integers(0, size - new_w + 1)),
                  int(rng.integers(0, size - new_h + 1)))
    canvas.paste(resized, offset)
    return canvas


def _rng() -> np.random.Generator:
    """A generator seeded from torch, so dataloader workers do not agree.

    numpy's global state is inherited by every worker process unchanged, so
    seeding from it would hand all four workers the same augmentation stream
    and silently cut the effective variety by a factor of four. torch reseeds
    each worker per epoch, which is the property needed here.
    """
    return np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))


def _smooth_field(rng, height: int, width: int, control: int = 6) -> np.ndarray:
    """A low-frequency field on roughly [-1, 1], from a coarse grid upsampled.

    Low frequency is the point. A per-pixel random field is noise, which the
    blur already covers; a field with a handful of control points produces
    variation on the scale of the otolith itself, which is the scale on which
    real specimens differ from one another.
    """
    coarse = rng.uniform(-1.0, 1.0, size=(control, control)).astype(np.float32)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def elastic_orthogonal(image: Image.Image, rng, amplitude: float = 0.05,
                       control: int = 6) -> Image.Image:
    """Deform the section across its long axis, leaving the long axis alone.

    This is the only geometric augmentation here that is label-preserving by
    construction rather than by assumption. Age is read as a count of increments
    along a transect from the nucleus to the margin, which runs along the
    otolith's long axis. A displacement field that is zero along that axis and
    smooth across it changes the outline of the section and the curvature of its
    bands without changing the spacing of anything that gets counted, so the
    correct answer cannot move. Displacing along the reading axis would stretch
    and compress increment spacing, which is the signal itself, and that is why
    the field is applied to one axis only rather than isotropically.
    """
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    field = _smooth_field(rng, h, w, control)
    if w >= h:
        ys = ys + amplitude * h * field      # long axis horizontal
    else:
        xs = xs + amplitude * w * field
    warped = cv2.remap(arr, xs, ys, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def jitter_crop_box(image: Image.Image, rng, fraction: float = 0.08) -> Image.Image:
    """Move each edge of the crop independently by up to `fraction` of its side.

    The crops arrive from an upstream segmentation step, and every training
    image is that step's single best answer, so the model never sees the
    variability it will actually meet in deployment. Edges that move inward
    crop; edges that move outward are filled by replicating the border, because
    the true surrounding pixels are gone once the crop was taken. Replication is
    the honest filler: it adds no structure that could be mistaken for otolith.
    """
    w, h = image.size
    left, right = (rng.uniform(-fraction, fraction, 2) * w).round().astype(int)
    top, bottom = (rng.uniform(-fraction, fraction, 2) * h).round().astype(int)
    arr = np.asarray(image)

    pad = [max(0, top), max(0, bottom), max(0, left), max(0, right)]
    if any(pad):
        arr = cv2.copyMakeBorder(arr, *pad, borderType=cv2.BORDER_REPLICATE)
    y0, x0 = max(0, -top), max(0, -left)
    y1 = arr.shape[0] - max(0, -bottom)
    x1 = arr.shape[1] - max(0, -right)
    arr = arr[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    return Image.fromarray(arr)


def illumination_field(image: Image.Image, rng, strength: float = 0.12,
                       control: int = 4) -> Image.Image:
    """Multiply by a smooth field, simulating uneven reflected light.

    Sections are photographed under reflected light across a slide holding
    several otoliths, so illumination varies across the frame and between
    slides. Multiplicative and low-frequency is the right shape for that:
    additive brightness jitter would move the whole image equally, which is not
    how a light source behaves.
    """
    arr = np.asarray(image).astype(np.float32)
    field = 1.0 + strength * _smooth_field(rng, arr.shape[0], arr.shape[1], control)
    arr = np.clip(arr * field[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def randomised_clahe(image: Image.Image, rng, clip_range=(1.5, 5.0),
                     tile_range=(15, 40)) -> Image.Image:
    """CLAHE with its parameters drawn per image rather than fixed.

    The fixed clip limit of 3.0 tunes the encoder to one enhancement setting.
    Sampling it makes the encoder invariant to the setting instead, and it
    perturbs local contrast the same way acquisition does, which generic
    brightness jitter does not.
    """
    return clahe_enhancement(
        image,
        clip_limit=float(rng.uniform(*clip_range)),
        tile_size=int(rng.integers(*tile_range)),
        repeat_clahe=False,
    )


def strong_augment(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    """The combined augmentation set, applied in acquisition order.

    Ordering is not arbitrary, and getting it wrong is not a subtle error. The
    sequence follows the physical pipeline: the specimen has a shape, the
    segmentation cuts a box around it, the lamp illuminates it unevenly, the
    camera records it, and only then does CLAHE run in software.

    Illumination therefore has to precede CLAHE. Applying it afterwards was
    tried first and visibly destroyed the signal: these sections are already
    close to saturation, so a multiplicative field applied to an
    already-equalised image clips the bright regions to white and takes the
    outer increments with them, which are the ones that separate an eight from
    a nine. In the correct order CLAHE sees the uneven illumination and locally
    equalises it away, which is precisely the invariance this augmentation is
    meant to train.

    These are tested together rather than one at a time on purpose. Each is
    worth well under the 1.6-point seed spread on its own, so a four-way screen
    would mostly measure which condition drew the luckiest seeds; the
    combination is the one comparison with enough expected effect to resolve.
    """
    rng = _rng()
    image = jitter_crop_box(image, rng)
    image = elastic_orthogonal(image, rng)
    image = illumination_field(image, rng)
    if not config.apply_clahe:
        return image.convert("RGB")
    return randomised_clahe(image, rng)


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
    strong = config.augment == "strong"

    # Under `strong` the contrast step is drawn per image rather than fixed, and
    # it runs after the geometric augmentations rather than before, so that the
    # enhancement is computed on the geometry the encoder actually receives.
    first = (transforms.Lambda(lambda img: strong_augment(img, config)) if strong
             else transforms.Lambda(lambda img: apply_config(img, config)))
    # Random placement only makes sense while there is empty canvas to place
    # into, which is to say under `pad`.
    place = (transforms.Lambda(
        lambda img: fit_to_square(img, size, config.resize_mode, rng=_rng()))
        if strong and config.resize_mode == "pad" else
        transforms.Lambda(lambda img: fit_to_square(img, size, config.resize_mode)))

    return transforms.Compose([
        first,
        transforms.RandomHorizontalFlip(p=hflip_prob),
        # Rotation happens before the square fit, on the original geometry, and
        # fills exposed corners with mid-grey rather than black so the encoder is
        # not handed a hard synthetic edge.
        transforms.RandomAffine(degrees=rotation_degrees, scale=scale_range, fill=128),
        place,
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=blur_prob
        ),
        transforms.ToTensor(),
        _normalization(processor, image_mean, image_std),
    ])
