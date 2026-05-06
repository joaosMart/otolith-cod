---
status: pending
priority: p2
issue_id: "009"
tags: [code-review, dependencies, packaging]
dependencies: []
---

# Missing scipy Dependency in pyproject.toml

## Problem Statement

The `compare_models_significance()` function imports `scipy.stats`, but `scipy` is not listed in `pyproject.toml` dependencies. The import will fail when the function is called.

## Findings

**Python Reviewer Agent:**
- Location: `src/evaluation/metrics.py:210`
- Late import inside function: `from scipy import stats`
- `pyproject.toml:12-36` does not include scipy

**Architecture Strategist Agent:**
- This is a hidden dependency
- Will cause runtime failure on first use

## Proposed Solutions

### Option A: Add scipy to Dependencies (Recommended)
**Pros:** Correct fix
**Cons:** Adds ~60MB dependency
**Effort:** Small (2 minutes)
**Risk:** None

```toml
# pyproject.toml
dependencies = [
    ...
    "scipy>=1.10.0",
]
```

### Option B: Make scipy Optional
**Pros:** Smaller base install
**Cons:** More complex, function may fail
**Effort:** Small
**Risk:** Low

```toml
[project.optional-dependencies]
stats = ["scipy>=1.10.0"]
```

## Recommended Action

[To be filled during triage]

## Technical Details

**Affected Files:**
- `pyproject.toml`
- `src/evaluation/metrics.py:210`

## Acceptance Criteria

- [ ] scipy added to dependencies
- [ ] `pip install -e .` includes scipy
- [ ] `compare_models_significance` works without errors

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Verify all imports are in dependencies |

## Resources

- pyproject.toml specification
