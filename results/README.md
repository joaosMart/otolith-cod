# Experiment results

Every number in the manuscript traces to a file in here. Each top-level
directory is one submitted batch (see `scripts/hf/submit.py`); within a batch,
`results/<condition>/bootstrap/results.json` holds the metrics and the exact
configuration that produced them, and `runs/<name>/training_info.json` holds the
fine-tuning history and measured parameter counts.

`summarize/paper_summary.json` is the consolidated view: per-age breakdown,
McNemar test, encoder and preprocessing comparisons, seed spread and attribution.
`summarize/paper_figures/` holds the figures as they appear in the paper.

## What is not here

Model weights and embedding caches are excluded to keep the repository
clonable. The complete archive, including LoRA adapters, lives at
`hafsteinn/otolith-cod-results` on the HuggingFace Hub. Embeddings are
regenerable in one to two minutes per encoder with `scripts/extract_embeddings.py`.

## Reading the condition names

    <encoder>-frozen                      frozen encoder
    <encoder>_lora_r<rank>a<alpha>_s<seed>_clahe    LoRA-adapted
    siglip2_lora-pool_...                 LoRA plus an unfrozen pooling head
    siglip2-frozen-{squash,crop}          alternative resize policies

The trailing `clahe` records the preprocessing configuration. Every setting that
changes a pixel appears in the name, so two configurations cannot collide.

## Known gaps

Three batches did not finish: full fine-tuning, the rank sweep and the learning
curves. The rank sweep and learning curves stopped mid-run with no error after
about 1h46m, matching a pattern in which no job has completed beyond 1h43m, so
they need to be resubmitted as shorter jobs. Full fine-tuning trained
successfully in 22 minutes but its extraction step failed: the batch definition
assumed a LoRA adapter directory, which that mode does not produce.
