# D32 — Minority-class (wheeze/both) error analysis, primary test set

Same frozen E01 predictions used for D29 (no retraining). 144 of 179 true
wheeze/both test cycles were misclassified (support: wheeze=106, both=73).
References are metadata only (`cycle_id`/`record_id`/device/chest_site) —
no raw audio is reproduced here, consistent with this project's data policy.

## Errors by device (absolute counts, not normalized by per-device support)

| true label | device | n_errors |
|---|---|---:|
| both | AKGC417L | 10 |
| both | Litt3200 | 6 |
| both | Meditron | 33 |
| wheeze | AKGC417L | 27 |
| wheeze | Litt3200 | 17 |
| wheeze | LittC2SE | 4 |
| wheeze | Meditron | 47 |

## Errors by chest site (absolute counts)

| true label | chest_site | n_errors |
|---|---|---:|
| both | Al | 3 |
| both | Ar | 14 |
| both | Ll | 1 |
| both | Lr | 6 |
| both | Pl | 10 |
| both | Pr | 14 |
| both | Tc | 1 |
| wheeze | Al | 11 |
| wheeze | Ar | 35 |
| wheeze | Ll | 8 |
| wheeze | Lr | 14 |
| wheeze | Pl | 10 |
| wheeze | Pr | 15 |
| wheeze | Tc | 2 |

**Caveat:** these are raw error counts, not normalized by how many wheeze/both
cycles each device/site actually has in the test set (that breakdown wasn't
computed here). Meditron and Ar/Pr dominate counts partly because they are
the most common device/site in this dataset generally (D05 audit) — this
table does **not** establish a device- or site-specific failure rate, only
where the raw error mass falls. A rate-normalized version would be needed
before claiming a device effect.

## 10 example references (error type breakdown)

| cycle_id | true | predicted | device | chest_site | error type |
|---|---|---|---|---|---|
| `112_1b1_Lr_sc_Meditron_c007` | wheeze | normal | Meditron | Lr | missed abnormal (false negative to Normal) |
| `201_1b3_Ar_sc_Meditron_c003` | wheeze | both | Meditron | Ar | confused with adjacent abnormal class |
| `218_1b1_Al_sc_Meditron_c004` | wheeze | both | Meditron | Al | confused with adjacent abnormal class |
| `112_1p1_Pr_sc_Litt3200_c007` | wheeze | normal | Litt3200 | Pr | missed abnormal (false negative to Normal) |
| `218_1b1_Ar_sc_Meditron_c013` | both | wheeze | Meditron | Ar | confused with adjacent abnormal class |
| `218_1b1_Pr_sc_Meditron_c008` | both | wheeze | Meditron | Pr | confused with adjacent abnormal class |
| `172_1b5_Pl_mc_AKGC417L_c004` | both | crackle | AKGC417L | Pl | confused with adjacent abnormal class |
| `112_1p1_Pr_sc_Litt3200_c002` | wheeze | normal | Litt3200 | Pr | missed abnormal (false negative to Normal) |
| `218_1b1_Pr_sc_Meditron_c009` | both | wheeze | Meditron | Pr | confused with adjacent abnormal class |
| `112_1p1_Ll_sc_Litt3200_c008` | wheeze | normal | Litt3200 | Ll | missed abnormal (false negative to Normal) |

## Error-type pattern

Across these 10 examples, two error types dominate: (1) **missed abnormal →
Normal** (4/10, all on `Litt3200`/`Meditron`), consistent with the model's
overall bias toward predicting Normal under noise (D31); (2) **confused
with an adjacent abnormal class** (6/10, mostly `wheeze↔both`), consistent
with `crackle`/`wheeze` co-occurring in the `both` label by definition —
the model is not randomly wrong, it is specifically bad at separating the
three abnormal classes from each other and from Normal, not failing
arbitrarily. This matches the standing `both`-class weakness flagged since
D21/D27 and carries forward to the model card (D34) as a named limitation,
not a surprise.
