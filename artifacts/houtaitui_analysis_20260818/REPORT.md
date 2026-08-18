# Houtaitui training comparison

Snapshot generated: `2026-08-18T10:59:31+08:00`

Values are means over the final configured summary window. Physical error tags are
per-step quantities and remain the safest metrics across reward/config changes.

## Directly comparable physical errors

| run | final iter | body pos | body rot | joint pos | anchor pos | anchor rot | joint vel |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-08-10_23-59-15_warmstart_from_0725_kneekd | 87498 | 0.0598 | 0.1967 | 0.9515 | 0.5676 | 0.0932 | 10.7251 |
| 2026-08-12_20-41-41 | 29997 | 0.0692 | 0.2158 | 1.0134 | 0.5866 | 0.1085 | 10.6532 |
| 2026-08-14_01-08-04 | 14981 | 0.0812 | 0.2489 | 1.1830 | 0.6256 | 0.1214 | 11.6717 |
| 2026-08-16_01-25-08 | 29998 | 0.1226 | 0.4255 | 2.3063 | 0.3021 | 0.2258 | 11.1069 |
| 2026-08-17_22-32-51 | 23552 | 0.1044 | 0.3758 | 2.0863 | 0.2546 | 0.1847 | 10.9546 |

## Termination shares

| run | anchor_ori | anchor_pos | anchor_pos_xy | ee_body_pos | motion_end | swing_foot_height | time_out |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-08-10_23-59-15_warmstart_from_0725_kneekd | 0.0% | 0.1% | - | 7.0% | - | - | 92.9% |
| 2026-08-12_20-41-41 | 0.1% | 0.2% | - | 8.7% | - | - | 91.0% |
| 2026-08-14_01-08-04 | 0.0% | 0.1% | - | 11.8% | - | - | 88.1% |
| 2026-08-16_01-25-08 | 0.0% | 0.0% | 71.3% | 1.3% | 26.9% | 0.4% | 0.0% |
| 2026-08-17_22-32-51 | 0.0% | 0.0% | 52.2% | 0.5% | 46.5% | 0.8% | 0.0% |

## Episode-normalized view

| run | mean length | estimated ceiling | length/ceiling | mean reward | reward/step |
|---|---:|---:|---:|---:|---:|
| 2026-08-10_23-59-15_warmstart_from_0725_kneekd | 1447.2 | 1500.0 | 0.965 | 230.35 | 0.1592 |
| 2026-08-12_20-41-41 | 1430.8 | 1500.0 | 0.954 | 221.27 | 0.1547 |
| 2026-08-14_01-08-04 | 1408.0 | 1500.0 | 0.939 | 205.46 | 0.1459 |
| 2026-08-16_01-25-08 | 190.7 | 421.2 | 0.453 | 28.97 | 0.1519 |
| 2026-08-17_22-32-51 | 321.5 | 569.2 | 0.565 | 55.16 | 0.1716 |

## Files

- `summary.json`: machine-readable summaries, fingerprints and checkpoint inventories.
- `curves_windows.csv.gz`: mean/min/max/last for every scalar in fixed iteration windows.
- `configs/<run>/`: immutable configuration snapshots archived by each run.
- `motions/`: reference NPZ files available when this snapshot was created.

Do not compare raw episode reward or episode length across runs until checking the
motion file, end behavior, reset sampling and termination set in `summary.json`.
