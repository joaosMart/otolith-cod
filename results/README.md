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
clonable. The complete archive, including LoRA adapters and the fine-tuned
encoder, lives at `hafsteinn/otolith-cod-results` on the HuggingFace Hub.
Embeddings regenerate in one to two minutes per encoder with
`scripts/extract_embeddings.py`.

## Reading the condition names

    <encoder>-frozen                               frozen encoder
    <encoder>_lora_r<rank>a<alpha>_s<seed>_clahe   LoRA-adapted
    <encoder>_lora_r16a32_f<fraction>_s42_clahe    learning-curve point
    siglip2_full_s42_clahe                         full fine-tuning
    siglip2_lora-pool_...                          LoRA plus an unfrozen pooling head
    siglip2-frozen-{squash,crop}                   alternative resize policies

The trailing `clahe` records the preprocessing configuration. Every setting that
changes a pixel appears in the name, so two configurations cannot collide.

## A note on job sizing

Three batches initially failed. The rank sweep and the learning curves stopped
mid-run with no Python error at about 1h46m, matching a pattern in which nothing
had completed beyond 1h43m regardless of the timeout requested. Jobs on this
platform appear to be capped near two hours in practice. Splitting them into one
job per run fixed it, and every batch in `submit.py` is now sized to finish well
inside 110 minutes.

The full fine-tuning batch failed separately, for a reason of our own making:
that mode saves a state dict rather than a PEFT adapter directory, and the chain
builder was asking extraction for a path it never produces. Training itself had
succeeded in 22 minutes.

All batches have since completed.
