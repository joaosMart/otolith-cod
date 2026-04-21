---
status: pending
priority: p2
issue_id: "006"
tags: [code-review, security, reproducibility]
dependencies: []
---

# HuggingFace Model Loading Without Revision Pinning

## Problem Statement

Models are loaded from HuggingFace Hub without explicit revision pinning or checksum verification. This creates supply chain attack risk and reproducibility issues if models are updated upstream.

## Findings

**Security Sentinel Agent:**
- Location: `src/models/feature_extractor.py:53-58`
- No `revision` parameter specified
- No local caching with verification

```python
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    attn_implementation="sdpa",
)  # Missing revision=
```

## Proposed Solutions

### Option A: Pin Model Revisions (Recommended)
**Pros:** Reproducibility, security
**Cons:** Needs manual updates for new versions
**Effort:** Small (30 minutes)
**Risk:** None

```python
SUPPORTED_MODELS = {
    "clip-vit-l-14-336": ("openai/clip-vit-large-patch14-336", 768, "abc123hash"),
    "siglip2-so400m-14-384": ("google/siglip2-so400m-patch14-384", 1152, "def456hash"),
}

model = AutoModel.from_pretrained(model_id, revision=revision)
```

### Option B: Download and Cache Locally
**Pros:** Full control, offline capability
**Cons:** Storage overhead, manual management
**Effort:** Medium
**Risk:** Low

## Recommended Action

[To be filled during triage]

## Technical Details

**Affected Files:**
- `src/models/feature_extractor.py:53-58`

**Components:** Model loading

## Acceptance Criteria

- [ ] Model revisions pinned to specific commits
- [ ] Consider `local_files_only=True` for production
- [ ] Document model versions in WORKLOG.md

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Pin ML model versions for reproducibility |

## Resources

- [HuggingFace Model Versioning](https://huggingface.co/docs/transformers/main/en/installation#offline-mode)
