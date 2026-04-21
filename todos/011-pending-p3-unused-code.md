---
status: pending
priority: p3
issue_id: "011"
tags: [code-review, simplification, cleanup]
dependencies: []
---

# Unused Code and Over-Abstraction

## Problem Statement

The codebase contains significant unused code including entire classes, unused methods, and over-abstracted wrappers that add complexity without value.

## Findings

**Code Simplicity Reviewer Agent:**

### Unused Code (~400 lines)
| Location | Lines | Description |
|----------|-------|-------------|
| `dataset.py:189-232` | 43 | `get_class_weights()`, `get_sample_weights()` - never called |
| `dataset.py:234-259` | 25 | `subset()` method - never called |
| `dataset.py:274-350` | 77 | `OtolithEmbeddingDataset` class - never used |
| `shallow_models.py` | 151 | Entire file - all functions never used |
| `feature_extractor.py:122-167` | 45 | `extract_and_cache_features()`, `get_embedding_dim()` |
| `metrics.py` | ~70 | Trivial wrappers like `compute_accuracy()` |

### Estimated LOC Reduction: ~516 lines (40%)

## Proposed Solutions

### Option A: Gradual Cleanup (Recommended)
**Pros:** Safe, verifiable
**Cons:** Takes longer
**Effort:** Medium (2-3 hours)
**Risk:** Low

Clean up in order:
1. Remove `shallow_models.py` (use sklearn directly)
2. Remove `OtolithEmbeddingDataset` class
3. Remove unused dataset methods
4. Simplify metrics to 3 core functions

### Option B: Aggressive Cleanup
**Pros:** Fast
**Cons:** Risk of removing needed code
**Effort:** Medium
**Risk:** Medium

## Recommended Action

[To be filled during triage]

## Technical Details

**Files to Simplify:**
- `src/data/dataset.py` - Remove unused methods
- `src/models/shallow_models.py` - Delete entire file
- `src/evaluation/metrics.py` - Remove trivial wrappers

## Acceptance Criteria

- [ ] No dead code remains
- [ ] Scripts still work correctly
- [ ] LOC reduced by ~30%+

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Simpler code is easier to verify |

## Resources

- YAGNI principle
