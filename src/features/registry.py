"""
Single source of truth for the vision encoders under comparison.

Before this module, each encoder's peculiarities (which submodule PEFT attaches
to, what the attention projections are called, how to get a pooled vector out)
were spread across five hardcoded `if model == "siglip2" else ...` branches in
the extraction, fine-tuning and visualisation scripts. Adding a fifth encoder
meant finding all five. Now an encoder is one `EncoderSpec` entry and the scripts
never branch on model name.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPVisionModelWithProjection,
    SiglipVisionModel,
)


#: Vision-only loader classes. Loading the full vision-language model would pull
#: the text tower as well, roughly doubling the download and the GPU memory for
#: weights that are never used. Both classes still expose `.vision_model`, so the
#: readouts below are unchanged.
LOADER_CLASSES = {
    "CLIPVisionModelWithProjection": CLIPVisionModelWithProjection,
    "SiglipVisionModel": SiglipVisionModel,
    "AutoModel": AutoModel,
}


@dataclass(frozen=True)
class EncoderSpec:
    """Everything the pipeline needs to know about one vision encoder."""

    name: str
    model_id: str
    embedding_dim: int

    #: Module-name suffixes PEFT matches to inject LoRA. These differ by
    #: architecture: HuggingFace CLIP-family towers use separate q/k/v
    #: projections, while timm-style towers fuse them into a single `qkv`.
    lora_targets: Tuple[str, ...]

    #: How the pooled embedding is produced. See `embed_images`.
    readout: str

    #: Modules that a full-rank unfreeze would add on top of LoRA. Empty by
    #: default: the headline configuration is pure low-rank adaptation, and
    #: unfreezing these is an opt-in ablation, not the default behaviour.
    poolable_modules: Tuple[str, ...] = ()

    #: "auto" -> AutoProcessor, "image" -> AutoImageProcessor.
    processor_kind: str = "auto"

    #: Set when the pooled vector passes through a projection that lives outside
    #: the vision tower, so LoRA does not adapt it but extraction must apply it.
    projection_attr: Optional[str] = None

    #: Key into LOADER_CLASSES. Vision-only classes avoid downloading a text
    #: tower that is never used.
    loader_class: str = "AutoModel"

    image_size: Optional[int] = None
    notes: str = ""


ENCODERS = {
    "clip": EncoderSpec(
        name="clip",
        model_id="openai/clip-vit-large-patch14-336",
        embedding_dim=768,
        lora_targets=("q_proj", "k_proj", "v_proj", "fc1", "fc2"),
        readout="vision_pooler",
        poolable_modules=("post_layernorm",),
        processor_kind="auto",
        projection_attr="visual_projection",
        loader_class="CLIPVisionModelWithProjection",
        image_size=336,
        notes="Reference encoder used by Sigurdardottir et al. (2023).",
    ),
    "siglip2": EncoderSpec(
        name="siglip2",
        model_id="google/siglip2-so400m-patch14-384",
        embedding_dim=1152,
        lora_targets=("q_proj", "k_proj", "v_proj", "fc1", "fc2"),
        readout="vision_pooler",
        # `head` is the multi-head attention pooling module that produces the
        # image-level vector; SigLIP2 has no class token. Unfreezing it adds
        # 15.24M full-rank parameters against 7.70M of LoRA at rank 16.
        poolable_modules=("head", "post_layernorm"),
        processor_kind="auto",
        projection_attr=None,
        loader_class="SiglipVisionModel",
        image_size=384,
        notes="Strongest frozen encoder in the four-way comparison.",
    ),
    "dinov2": EncoderSpec(
        name="dinov2",
        model_id="facebook/dinov2-with-registers-large",
        embedding_dim=1024,
        lora_targets=("query", "key", "value", "fc1", "fc2"),
        readout="cls_token",
        poolable_modules=("layernorm",),
        processor_kind="image",
        projection_attr=None,
        image_size=518,
        notes="Self-supervised; 4 register tokens sit between CLS and patches.",
    ),
    "dinov3": EncoderSpec(
        name="dinov3",
        model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        embedding_dim=1024,
        # Verified against the checkpoint's state dict: separate q/k/v
        # projections and a gated MLP with up_proj / down_proj, unlike DINOv2's
        # query/key/value and fc1/fc2.
        lora_targets=("q_proj", "k_proj", "v_proj", "up_proj", "down_proj"),
        readout="cls_token",
        poolable_modules=("norm",),
        processor_kind="image",
        projection_attr=None,
        # Rotary position embeddings, so the input resolution is free rather than
        # fixed by a learned position table. Set to 384 to match SigLIP2's token
        # grid rather than DINOv2's 518, which upsampled almost every crop: the
        # short side of three quarters of these images is already below 384.
        image_size=384,
        notes=(
            "Weights are gated on the Hub and need manual approval. The paper "
            "must carry a 'Built with DINOv3' acknowledgment under its licence."
        ),
    ),
}


#: Number of leading non-patch tokens for encoders that prepend a class token
#: and register tokens. Used when reshaping tokens back onto the image grid.
PREFIX_TOKENS = {
    "clip": 1,
    "siglip2": 0,
    "dinov2": 5,  # 1 class token + 4 registers
    "dinov3": 5,  # same layout: 1 class token + 4 registers
}


def get_spec(name: str) -> EncoderSpec:
    if name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {', '.join(sorted(ENCODERS))}"
        )
    return ENCODERS[name]


def load_encoder(
    name: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, object, EncoderSpec]:
    """Load an encoder and its image processor.

    Returns the full model rather than the vision tower, because some readouts
    need a projection that lives outside the tower.
    """
    spec = get_spec(name)
    loader = LOADER_CLASSES[spec.loader_class]
    model = loader.from_pretrained(spec.model_id, torch_dtype=dtype)

    if spec.processor_kind == "image":
        processor = AutoImageProcessor.from_pretrained(spec.model_id)
    else:
        processor = AutoProcessor.from_pretrained(spec.model_id)

    if device is not None:
        model = model.to(device)

    return model, processor, spec


def adaptable_module(model: torch.nn.Module, spec: EncoderSpec) -> torch.nn.Module:
    """The submodule LoRA should be injected into.

    For vision-language models this is the vision tower alone, so the text tower
    is left untouched and is not carried around in the saved adapter.
    """
    if spec.readout in ("vision_pooler",):
        return model.vision_model
    return model


def embed_images(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    spec: EncoderSpec,
    normalize: bool = True,
) -> torch.Tensor:
    """Produce one pooled embedding per image.

    Differentiable, so the same function serves feature extraction and the
    forward pass during fine-tuning. Using one readout for both is what makes the
    frozen and adapted conditions comparable: any difference between them is the
    encoder weights, never the pooling.
    """
    if spec.readout == "vision_pooler":
        outputs = model.vision_model(pixel_values=pixel_values)
        features = outputs.pooler_output
        if spec.projection_attr is not None:
            features = getattr(model, spec.projection_attr)(features)
    elif spec.readout == "cls_token":
        outputs = model(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
    else:
        raise ValueError(f"Unhandled readout '{spec.readout}' for encoder {spec.name}")

    if normalize:
        features = F.normalize(features, p=2, dim=-1, eps=1e-8)
    return features


def patch_tokens(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    spec: EncoderSpec,
) -> torch.Tensor:
    """Per-patch token representations, with class and register tokens stripped.

    Used by the attention and saliency visualisations, where including the
    register tokens in a spatial map would misalign the grid.
    """
    if spec.readout == "vision_pooler":
        hidden = model.vision_model(pixel_values=pixel_values).last_hidden_state
    else:
        hidden = model(pixel_values=pixel_values).last_hidden_state

    prefix = PREFIX_TOKENS.get(spec.name, 0)
    return hidden[:, prefix:, :]


def count_parameters(module: torch.nn.Module) -> dict:
    """Break trainable parameters into low-rank and full-rank portions.

    Reported in the paper, so it is computed from the live model rather than
    derived by hand. Getting this wrong is exactly how the manuscript ended up
    quoting a parameter count that was too low.
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    lora = sum(
        p.numel()
        for n, p in module.named_parameters()
        if p.requires_grad and "lora_" in n
    )
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "full_rank_trainable": trainable - lora,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
        "lora_pct": 100.0 * lora / total if total else 0.0,
    }
