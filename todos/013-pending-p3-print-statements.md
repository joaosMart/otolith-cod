---
status: pending
priority: p3
issue_id: "013"
tags: [code-review, logging, quality]
dependencies: []
---

# Print Statements Instead of Logging

## Problem Statement

The codebase uses 59 `print()` statements instead of Python's logging module, despite having a logging configuration section in `config.yaml` that is never used.

## Findings

**Multiple Agents Identified:**
- Config defined at `configs/config.yaml:80-84` but unused
- 59 print() calls across 4 files
- No way to control verbosity or redirect output

```yaml
# Defined but unused
logging:
  level: "INFO"
  save_to_file: true
  log_dir: "outputs/logs"
```

## Proposed Solutions

### Option A: Implement Proper Logging (Recommended)
**Pros:** Configurable, professional
**Cons:** Refactoring effort
**Effort:** Medium (1-2 hours)
**Risk:** Low

```python
import logging
logging.basicConfig(
    level=config["logging"]["level"],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
```

### Option B: Keep Print, Remove Config
**Pros:** Simple
**Cons:** Less professional
**Effort:** Small
**Risk:** None

## Recommended Action

[To be filled during triage]

## Technical Details

**Affected Files:**
- `scripts/run_experiment.py`
- `scripts/train_shallow_classifier.py`
- `src/models/feature_extractor.py`
- `src/data/dataset.py`

## Acceptance Criteria

- [ ] All print() replaced with logger calls
- [ ] Logging config from YAML is used
- [ ] Log level can be controlled via config

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-16 | Identified during code review | Use logging module for production code |

## Resources

- Python logging documentation
