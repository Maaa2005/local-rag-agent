# OOD-targeted remediation: previous FT vs new FT

The fixed 30-record human-authored OOD set was never included in training. New training prompts had maximum character-trigram similarity 0.1667 to OOD (threshold 0.30).

| OOD metric | Previous FT | New FT | Delta |
|---|---:|---:|---:|
| JSON parse | 27/30 (90.0%) | 29/30 (96.7%) | +6.7 pp |
| Schema valid | 19/30 (63.3%) | 28/30 (93.3%) | +30.0 pp |
| Status | 16/30 (53.3%) | 24/30 (80.0%) | +26.7 pp |
| Intent | 16/30 (53.3%) | 23/30 (76.7%) | +23.3 pp |
| Action | 15/30 (50.0%) | 24/30 (80.0%) | +30.0 pp |
| Clarification decision | 19/30 (63.3%) | 26/30 (86.7%) | +23.3 pp |
| Rejection (overall) | 25/30 (83.3%) | 27/30 (90.0%) | +6.7 pp |
| Question fields exact | 10/30 (33.3%) | 19/30 (63.3%) | +30.0 pp |
| Question fields F1 (mean) | 36.3% | 71.3% | +35.0 pp |
| Safe search filter | 2/8 (25.0%) | 8/8 (100.0%) | +75.0 pp |

## Target-group diagnostics

| Group | Previous status | New status | Previous schema | New schema | Strict outcome |
|---|---:|---:|---:|---:|---:|
| needs_clarification | 9/10 | 7/10 | 8/10 | 10/10 | Question fields exact: 0/10 -> 0/10 |
| ready | 2/10 | 10/10 | 8/10 | 9/10 | Status: 2/10 -> 10/10 |
| rejected | 5/10 | 7/10 | 3/10 | 9/10 | Rejection (overall): 5/10 -> 7/10 |

## Retained held-outs

- Validation: 103 targets, all reported metrics 100%
- Test: 124 targets, all reported metrics 100%

## Case-level comparison

- Core-perfect fixes: 9 (human-ood-006, human-ood-011, human-ood-014, human-ood-015, human-ood-019, human-ood-021, human-ood-022, human-ood-026, human-ood-029)
- Core regressions: 1 (human-ood-010)

## Conclusion

The new FT substantially improves ready-policy handling, schema compliance, safe filters, Windows-path JSON, and diverse refusal behavior without regressing the retained validation/test sets. Remaining weaknesses are exact clarification slot naming (still 0/10), three clarification decisions incorrectly treated as ready, two unsafe requests not rejected, one prompt-injection response emitted outside JSON, and a few intent-category disagreements.
