---
status: pending
priority: p1
issue_id: "003"
tags: [code-review, quality, debug-code]
dependencies: []
---

# Debug/Testing Code Left in Production Script

## Problem Statement

Hardcoded debug code in `train_shallow_classifier.py` permanently filters the data to ages 1-10, regardless of user intent. This code is marked `## TESTING ##` and silently corrupts results.

## Findings

**Multiple Agents Identified This Issue:**

**Pattern Recognition Agent:**
- Location: `scripts/train_shallow_classifier.py:156-169`
- Marked with `## TESTING ##` comment
- Hardcodes `target_ages = [1,2,3,4,5,6,7,8,9,10]`

**Code Simplicity Reviewer:**
- This is clearly leftover debug code
- Should use config's `age_range` parameter if filtering is needed

**Python Reviewer:**
- Severity: CRITICAL
- Silent data corruption

## Proposed Solutions

### Option A: Remove Entirely (Recommended)
**Pros:** Clean, simple
**Cons:** None - this is debug code
**Effort:** Small (2 minutes)
**Risk:** None

Delete lines 156-169.

### Option B: Make Configurable
**Pros:** Preserves functionality if needed
**Cons:** Adds complexity for debugging feature
**Effort:** Medium
**Risk:** Low

Add `--filter-ages` CLI argument.

## Recommended Action

[To be filled during triage]

## Technical Details

**Affected Files:**
- `scripts/train_shallow_classifier.py:156-169`

**Code to Remove:**
```python
##########
## TESTING
##########

target_ages = [1,2,3,4,5,6,7,8,9,10]
mask = np.isin(labels, target_ages)
filter_features = features[mask]
filter_labels = labels[mask]
```

## Acceptance Criteria

- [ ] Debug code removed from production script
- [ ] Verify script works correctly after removal
- [ ] No other debug code blocks exist in codebase

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Always remove debug code before committing |

## Resources

- PR context: Migration from OpenCLIP to HuggingFace
