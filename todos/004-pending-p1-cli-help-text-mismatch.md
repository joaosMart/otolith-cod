---
status: pending
priority: p1
issue_id: "004"
tags: [code-review, documentation, usability]
dependencies: []
---

# CLI Argument Help Text Does Not Match Default Values

## Problem Statement

In `train_shallow_classifier.py`, the CLI argument help strings show different values than the actual defaults. Users will be confused and documentation is effectively lying.

## Findings

**Python Reviewer Agent:**
- 5 arguments with mismatched help/defaults

| Argument | Actual Default | Help Says |
|----------|----------------|-----------|
| `--train-ratio` | 0.8 | 0.65 |
| `--val-ratio` | 0.1 | 0.15 |
| `--test-ratio` | 0.1 | 0.20 |
| `--alpha-steps` | 100 | 20 |
| `--cv-folds` | 10 | 5 |

**Location:** `scripts/train_shallow_classifier.py:62-101`

## Proposed Solutions

### Option A: Update Help Strings (Recommended)
**Pros:** Quick fix
**Cons:** None
**Effort:** Small (5 minutes)
**Risk:** None

### Option B: Update Defaults to Match Help
**Pros:** If original help values were intentional
**Cons:** Changes behavior
**Effort:** Small
**Risk:** Medium - may affect experiments

## Recommended Action

[To be filled during triage]

## Technical Details

**Affected Files:**
- `scripts/train_shallow_classifier.py:62-101`

**Fixes Required:**
```python
parser.add_argument("--train-ratio", default=0.8, help="Training set ratio (default: 0.8)")
parser.add_argument("--val-ratio", default=0.1, help="Validation set ratio (default: 0.1)")
parser.add_argument("--test-ratio", default=0.1, help="Test set ratio (default: 0.1)")
parser.add_argument("--alpha-steps", default=100, help="Number of alpha values (default: 100)")
parser.add_argument("--cv-folds", default=10, help="Number of CV folds (default: 10)")
```

## Acceptance Criteria

- [ ] All CLI help strings match actual default values
- [ ] Run `--help` and verify accuracy
- [ ] Consider generating help from config file

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Keep help strings synchronized with defaults |

## Resources

- Python argparse documentation
