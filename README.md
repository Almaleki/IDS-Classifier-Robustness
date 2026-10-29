# IDS Classifier-Level Numerical Robustness

This repository accompanies a study of numerical adversarial
robustness on CSE-CIC-IDS2018. It compares a baseline neural
network with FGSM-trained and PGD-trained networks across
five training seeds.

The attacks modify standardized numerical flow features.
They do not generate or validate executable network traffic.

## Publicly available materials

The repository includes training and evaluation scripts,
experiment configuration, aggregate results, and the
original balanced partition's split indices and fitted
preprocessing object.

Model checkpoints, the prepared balanced CSV, and the
supplementary grouped-run artifacts are not included.
Evaluation scripts requiring these artifacts cannot be
run directly from the repository alone. Readers can
generate new artifacts through the training workflow below.

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
