---
status: pending
priority: p2
issue_id: "008"
tags: [code-review, architecture, dry]
dependencies: []
---

# Code Duplication Across Multiple Locations

## Problem Statement

Several pieces of functionality are duplicated between library code and scripts, creating maintenance burden and inconsistency risks.

## Findings

**Pattern Recognition Agent + Architecture Strategist:**

### 1. Feature Caching Logic (Duplicated)
- Library: `src/models/feature_extractor.py:122-159` (`extract_and_cache_features`)
- Script: `scripts/run_experiment.py:69-108` (`extract_features_for_model`)

### 2. Class Weight Computation (Duplicated)
- `src/data/dataset.py:189-214` (`OtolithDataset.get_class_weights`)
- `src/data/dataset.py:338-349` (`OtolithEmbeddingDataset.get_class_weights`)

### 3. Metrics Functions (Overlapping)
- `src/evaluation/metrics.py:93-114` (`compute_classification_metrics`)
- `src/evaluation/metrics.py:117-132` (`compute_all_metrics`) - strict subset

## Proposed Solutions

### Option A: Consolidate Duplicates (Recommended)
**Pros:** DRY, easier maintenance
**Cons:** Requires refactoring
**Effort:** Medium (2-3 hours)
**Risk:** Low

Actions:
1. Remove `extract_features_for_model` from script, use library function
2. Extract `compute_class_weights` to shared utility
3. Remove `compute_all_metrics`, use `compute_classification_metrics`

### Option B: Document and Accept
**Pros:** No code changes
**Cons:** Technical debt remains
**Effort:** Small
**Risk:** Medium - divergence over time

## Recommended Action

[To be filled during triage]

## Technical Details

**Affected Files:**
- `src/models/feature_extractor.py`
- `scripts/run_experiment.py`
- `src/data/dataset.py`
- `src/evaluation/metrics.py`

## Acceptance Criteria

- [ ] No duplicate caching logic exists
- [ ] Single class weights utility function
- [ ] Single metrics computation function
- [ ] All tests pass after refactoring

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Keep utility functions in library, not scripts |

## Resources

- DRY Principle
