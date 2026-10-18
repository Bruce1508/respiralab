# Official ICBHI split audit (D06)

Source: `ICBHI_challenge_train_test.txt` (official page, downloaded directly;
920 rows, one per recording — matches the full dataset).

## Verdict
**NOT patient-disjoint.** This project's primary evaluation protocol (D07) uses its own
frozen patient-disjoint split regardless of this result — the official split
is reported here as a separate, audited comparison protocol only
(docs/03_DATA_AND_EVALUATION_CONTRACT.md), never combined with primary-split
numbers in one leaderboard.

## Counts
- Rows parsed: 920
- Unique train patients: 79
- Unique test patients: 49
- Patients appearing in **both** train and test: 2

## Secondary-source claim check (patient IDs 156, 218)
- Confirmed overlapping in the actual official file: ['156', '218']
- Claimed but NOT found overlapping here: none
- Additional overlapping patients beyond the two claimed: none

## All overlapping patient IDs
| patient_id |
|---|
| 156 |
| 218 |
