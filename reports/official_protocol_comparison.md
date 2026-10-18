# D30 - Official ICBHI split comparison (separate protocol)

**WARNING - this is NOT the primary evaluation.** The official ICBHI
train/test split is confirmed NOT patient-disjoint (D06):
['156', '218'] appear in BOTH official train
and test (120 of 2756 official-test cycles
belong to an overlapping patient). This number must never be combined with
D29's primary-split result in one leaderboard, per
docs/03_DATA_AND_EVALUATION_CONTRACT.md.

Evaluated at commit `c30ccec02109035190b1706c3e7a19537416a663`. Same frozen E01 pipeline (StandardScaler +
LogisticRegression class_weight=balanced) as D29, refit on the OFFICIAL
train rows only, evaluated on the OFFICIAL test rows only - entirely
separate fit from the primary-split model.

## Official-split counts
- Official train: 4131 cycles, 79 patients
- Official test: 2756 cycles, 49 patients
- Overlapping patients (in both): ['156', '218']

## Results (official split, NOT leakage-free)
- macro_f1 = 0.3366
- icbhi_score = 0.3919 (Sp=0.2774, Se=0.5064)
- per_class: {
  "normal": {
    "precision": 0.7075928917609047,
    "recall": 0.2773907536415453,
    "f1": 0.39854413102820746,
    "support": 1579
  },
  "crackle": {
    "precision": 0.4010989010989011,
    "recall": 0.5624036979969184,
    "f1": 0.46824887748556765,
    "support": 649
  },
  "wheeze": {
    "precision": 0.22469135802469137,
    "recall": 0.4727272727272727,
    "f1": 0.30460251046025105,
    "support": 385
  },
  "both": {
    "precision": 0.11750599520383694,
    "recall": 0.34265734265734266,
    "f1": 0.175,
    "support": 143
  }
}

## Comparability warning
This official-split macro_f1 (0.3366) is **not directly
comparable** to D29's primary-split macro_f1 (0.2861) - different train set,
different test set, and the official test set has 2 of its patients also
present in official train, inflating this number by an unknown amount. Report
both numbers separately, always labeled by protocol, never averaged or
leaderboarded together.
