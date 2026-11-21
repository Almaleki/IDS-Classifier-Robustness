# IDS Classifier-Level Numerical Robustness

This repository accompanies a study of numerical adversarial
robustness on CSE-CIC-IDS2018. It compares a baseline neural
network with FGSM-trained and PGD-trained networks across
five training seeds.

The attacks modify standardized numerical flow features.
They do not generate or validate executable network traffic.

## Publicly available materials

The repository contains the training and evaluation scripts, experiment
configuration, per-seed metrics, split indices, and fitted preprocessing
object for the original partition.

Download the following files from [release v1.0.0](https://github.com/Almaleki/IDS-Classifier-Robustness/releases/tag/v1.0.0):

| File | Contents |
|---|---|
| `balanced_dataset_verified.zip` | The exact balanced CSV used for both partitions, with all 971,604 records in the original row order. |
| `original_checkpoints_20260919_164823_076724.zip` | All 15 original-partition checkpoints for the baseline, FGSM-trained, and PGD-trained models across seeds 42–46. |
| `grouped_run_20261028_001334_956812.zip` | All 15 grouped-partition checkpoints, grouped split indices, group identifiers, fitted preprocessing object, and per-seed metrics for the primary and Auto-PGD evaluations. |

Before applying either set of split indices, verify the SHA-256 of the
extracted `multiclass_balanced.csv`:

`55b70c912a14cadd6760d2f7f4874837c4af0c8efacdb2e8b9adac09ea37e21f`

The balanced CSV was derived from [CSE-CIC-IDS2018](https://registry.opendata.aws/cse-cic-ids2018/),
produced by the Communications Security Establishment and the Canadian
Institute for Cybersecurity.

## Dataset preparation

Obtain CSE-CIC-IDS2018 from:
https://www.unb.ca/cic/datasets/ids-2018.html

The experiment used merged flow CSV files with the following
mapping:

- Benign → Benign
- FTP-BruteForce, SSH-Bruteforce, Brute Force -Web,
  Brute Force -XSS → BruteForce
- DoS attacks-GoldenEye, DoS attacks-Slowloris,
  DoS attacks-SlowHTTPTest, DoS attacks-Hulk → DoS
- DDoS attacks-LOIC-HTTP, DDOS attack-LOIC-UDP,
  DDOS attack-HOIC → DDoS
- Bot → Botnet
- Infilteration → Infiltration

SQL Injection and unmapped labels were excluded.
Store the mapped labels in an `AttackCategory` column.

For each retained class, sample 161,934 records without
replacement using pandas `sample(..., random_state=42)`.
The resulting balanced dataset contains 971,604 records.
Save it as `multiclass_balanced.csv`.

The exact sampled records depend on the merged source-row
order. Saved split indices apply only to the corresponding
balanced CSV in its original row order. Check its SHA256
against `dataset_sha256` in `preprocessing.joblib` before
using the published partition. Do not apply those indices
to a differently ordered or independently sampled CSV.

## Dependencies

The original configuration records:
- PyTorch 2.8.0
- NumPy 2.0.2
- pandas 2.3.3
- scikit-learn 1.6.1

The scripts also require SciPy, matplotlib, and joblib.
The standard Auto-PGD evaluation requires
adversarial-robustness-toolbox 1.20.1.

Install packages in the Python environment used to execute
the scripts. Consult `configuration.json` for the recorded
original environment. Newly generated runs record their
own configuration; identical numerical results across
different software and hardware environments are not
guaranteed.

## Execution order

1. Edit the dataset and output paths in `FullExpV5.py`.
   Set `RUN_UNBALANCED = False` to run only the three
   balanced configurations described here.
   Run the script to generate trained checkpoints,
   preprocessing, split indices, and primary results.

2. Point `Exp1_Overlap_Audit.py` to that run directory
   and its exact dataset. Run it to measure exact and
   near-duplicate overlap.

3. Point `Exp15_Stronger_attack_evaluation.py` to the
   same run and dataset. Check its command-line options
   using `--help`. This script evaluates saved models.

4. Point `Exp1_Grouped_Overlap_Control.py` to the
   original run and matching balanced CSV. Run it to
   create the supplementary grouped partition and
   train its three configurations across five seeds.

5. Point `Exp1_ART_AutoPGD.py` to the newly generated
   grouped-run directory and matching CSV. Run it to
   evaluate the supplementary checkpoints using ART
   Auto-PGD with CE and DLR objectives.

`Day_Based_Split.py` provides a timestamp/class-distribution
diagnostic; it does not train a day-disjoint model.

## Path configuration

Several scripts retain absolute paths from the author's
workstation. Replace them with local paths before running.
Check `RUN_DIR`, `ORIGINAL_RUN`, `RUN`, `BASE`,
`BALANCED_DATASET`, `CSV`, and output-directory settings
where present.

Training creates timestamped output folders. Update later
scripts to reference the actual generated folder. Preserve
the generated folder structure, feature order, class
mapping, and associated preprocessing.

## Interpretation

Five training seeds measure training variability within
each fixed partition. The supplementary grouping removes
exact feature-vector overlap, but does not enforce
capture-day, host, or session separation.

The ART evaluation uses Auto-PGD components, not the
complete AutoAttack ensemble. Report results from the
original and supplementary partitions separately.
