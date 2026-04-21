---
status: pending
priority: p2
issue_id: "010"
tags: [code-review, quality, error-handling]
dependencies: []
---

# Using Assert for Runtime Validation

## Problem Statement

The `create_train_val_test_splits()` function uses `assert` for validating that ratios sum to 1.0. Assertions can be disabled with `python -O`, which would silently skip this important validation.

## Findings

**Architecture Strategist Agent + Python Reviewer Agent:**
- Location: `src/data/splits.py:123-125`

```python
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
    f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
)
```

## Proposed Solutions

### Option A: Replace with ValueError (Recommended)
**Pros:** Cannot be disabled, proper error handling
**Cons:** None
**Effort:** Small (2 minutes)
**Risk:** None

```python
if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
    raise ValueError(
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )
```

## Recommended Action

[To be filled during triage]

## Technical Details

**Affected Files:**
- `src/data/splits.py:123-125`

## Acceptance Criteria

- [ ] Assert replaced with explicit ValueError
- [ ] Error message preserved
- [ ] Function still validates input correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Use exceptions, not asserts, for validation |

## Resources

- Python assert vs exceptions best practices
