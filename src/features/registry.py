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


class _TimmProcessorShim:
    """Presents timm's data config the way the transform builders expect it.

    timm has no image-processor object; it exposes resolution and normalisation
    statistics as a plain dict. Rather than teach the preprocessing module about
    a second convention, this adapts timm to the one it already uses.
    """

    def __init__(self, data_config):
        size = data_config["input_size"][-1]
        self.size = {"height": size, "width": size}
        self.image_mean = list(data_config["mean"])
        self.image_std = list(data_config["std"])


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

    #: "image" -> AutoImageProcessor, "auto" -> AutoProcessor. Prefer "image":
    #: the full processor also loads a text tokenizer, which no vision-only
    #: path uses and which costs a 34 MB download per job.
    processor_kind: str = "image"

    #: Set when the pooled vector passes through a projection that lives outside
    #: the vision tower, so LoRA does not adapt it but extraction must apply it.
    projection_attr: Optional[str] = None

    #: Key into LOADER_CLASSES. Vision-only classes avoid downloading a text
    #: tower that is never used.
    loader_class: str = "AutoModel"

    image_size: Optional[int] = None

    #: Largest per-step batch that fits on a 44 GB card. Memory scales with the
    #: square of the token count, and the encoders differ a lot there: DINOv2 at
    #: 518px produces 1,373 tokens against SigLIP2's 729 at 384px. The training
    #: script divides the target batch by this and accumulates gradients, so the
    #: effective batch is identical across encoders and only the memory
    #: footprint changes.
    micro_batch: int = 32

    notes: str = ""


ENCODERS = {
    "clip": EncoderSpec(
        name="clip",
        model_id="openai/clip-vit-large-patch14-336",
        embedding_dim=768,
        lora_targets=("q_proj", "k_proj", "v_proj", "fc1", "fc2"),
        readout="vision_pooler",
        poolable_modules=("post_layernorm",),
        processor_kind="image",
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
        processor_kind="image",
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
        micro_batch=8,
        notes="Self-supervised; 4 register tokens sit between CLS and patches.",
    ),
    "dinov3": EncoderSpec(
        name="dinov3",
        # The timm mirror rather than facebook/dinov3-vitl16-pretrain-lvd1689m.
        # Identical weights under the same DINOv3 licence, but ungated: Meta
        # reviews access to their own repo by hand, which would block this run
        # and would also block any reviewer trying to reproduce it.
        model_id="hf-hub:timm/vit_large_patch16_dinov3.lvd1689m",
        embedding_dim=1024,
        # Verified by loading the checkpoint: this layout fuses the three
        # attention projections into one `qkv` matrix, so the split q/k/v names
        # used by the CLIP-family encoders match nothing here. `proj` is
        # deliberately excluded because it also matches `patch_embed.proj`,
        # which would adapt the patch embedding as well and break the parity
        # with the other encoders, where only attention and MLP are adapted.
        lora_targets=("qkv", "fc1", "fc2"),
        readout="timm_cls",
        poolable_modules=("norm",),
        processor_kind="timm",
        projection_attr=None,
        loader_class="timm",
        # Rotary position embeddings, so the input resolution is free rather
        # than fixed by a learned position table. Confirmed to run at 224, 256
        # and 384. Set to 384 to match SigLIP2's token grid rather than DINOv2's
        # 518, which upsampled almost every crop: the short side of three
        # quarters of these images is already below 384.
        image_size=384,
        notes=(
            "Built with DINOv3. The licence requires this acknowledgment in "
            "any publication using the weights."
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

    if spec.loader_class == "timm":
        import timm
        from timm.data import resolve_model_data_config

        # num_classes=0 strips the classifier; the readout below takes the class
        # token explicitly rather than timm's default average pooling, so that
        # DINOv3 is read the same way as DINOv2 and the comparison between the
        # two is about the weights and not about the pooling.
        model = timm.create_model(spec.model_id, pretrained=True, num_classes=0)
        model = model.to(dtype)
        processor = _TimmProcessorShim(resolve_model_data_config(model))
    else:
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
    is left untouched and is not carried around in the saved adapter. Some
    transformers releases wrap the tower in a `.vision_model` attribute and
    others expose it directly, so this resolves either shape rather than
    assuming one.
    """
    return getattr(model, "vision_model", model)


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

    The output field is resolved by inspecting what the model returned rather
    than by hardcoding a path per architecture. Model classes have been
    restructured between transformers releases, and a readout that assumes one
    layout fails at run time on the other, which is exactly the kind of breakage
    that is expensive to discover on rented hardware.
    """
    if spec.readout == "timm_cls":
        # timm models take a positional tensor and expose tokens through
        # forward_features. Index 0 is the class token; indices 1 to 4 are
        # register tokens.
        features = model.forward_features(pixel_values)[:, 0, :]
        if normalize:
            features = F.normalize(features, p=2, dim=-1, eps=1e-8)
        return features

    outputs = model(pixel_values=pixel_values)

    if spec.readout == "cls_token":
        features = outputs.last_hidden_state[:, 0, :]
    elif getattr(outputs, "image_embeds", None) is not None:
        # CLIPVisionModelWithProjection: already through the visual projection.
        features = outputs.image_embeds
    elif getattr(outputs, "pooler_output", None) is not None:
        # SigLIP2: the attention-pooling head's output. No class token exists.
        features = outputs.pooler_output
        if spec.projection_attr is not None:
            features = getattr(model, spec.projection_attr)(features)
    else:
        features = outputs.last_hidden_state[:, 0, :]

    if features.shape[-1] != spec.embedding_dim:
        raise RuntimeError(
            f"{spec.name} produced a {features.shape[-1]}-d embedding but the "
            f"registry declares {spec.embedding_dim}. The readout and the "
            "checkpoint disagree; do not trust downstream results."
        )

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
    if spec.readout == "timm_cls":
        hidden = model.forward_features(pixel_values)
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
