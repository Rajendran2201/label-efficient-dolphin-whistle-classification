# Label-Efficient Dolphin Whistle Classification

> **A comparative study of convolutional networks, general pretrained audio representations, and OpenWhistle-specific self-supervised representations under limited labeled data.**

This repository investigates how different audio-modeling approaches compare in ten-class dolphin whistle classification when labeled training data is limited. The project is designed around controlled comparisons, label-efficiency experiments, and robustness analysis.

**Project status:** Research in progress. Experimental results will be added after the evaluation protocol is completed.

---

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Hypotheses](#hypotheses)
- [Proposed Contributions](#proposed-contributions)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Evaluation Protocol](#evaluation-protocol)
- [Label-Efficiency Study](#label-efficiency-study)
- [Robustness Analysis](#robustness-analysis)
- [Implementation Roadmap](#implementation-roadmap)
- [Technology Stack](#technology-stack)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Limitations and Scope](#limitations-and-scope)
- [References](#references)

## Overview

Dolphin whistles are acoustic signals with structures that can be investigated using machine learning. Classifying these signals presents several challenges, including variation in recording quality, differences between whistle categories, and limited labeled examples for some classes.

This project compares three principal approaches:

1. **A CNN trained from scratch** using log-Mel spectrograms.
2. **A general pretrained audio encoder** evaluated with a frozen encoder and, where feasible, end-to-end fine-tuning.
3. **An OpenWhistle-specific Wav2Vec 2.0 model** evaluated using frozen representations and fine-tuning where compute permits.

The main focus is not simply to identify the model with the highest full-data score. Instead, the project measures how performance changes as the labeled training set becomes smaller, while keeping the evaluation protocol consistent.

> **Research integrity:** The hypotheses below are testable claims, not assumed outcomes. Results and conclusions will be reported only after experiments have been run.

## Research Question

> How do spectrogram-based convolutional networks and pretrained audio representations compare in ten-class dolphin whistle classification when the amount of labeled training data is limited?

## Hypotheses

- **H1 — Representation quality:** Pretrained audio encoders will outperform a CNN trained from scratch when labeled data is scarce.
- **H2 — Label efficiency:** The relative advantage of pretrained representations will be larger at 10–25% of the labeled training set than at 100%.
- **H3 — Robustness:** Performance will vary with signal-to-noise ratio (SNR), and models will differ in how well they classify lower-quality recordings.

These hypotheses will be evaluated empirically. A result that does not support a hypothesis will be reported as such.

## Proposed Contributions

The planned contributions are:

1. A controlled comparison of a scratch-trained CNN, a general pretrained audio encoder, and an OpenWhistle-specific encoder.
2. Learning curves across multiple labeled-data budgets.
3. Analysis of performance across SNR ranges and whistle categories, where reliable metadata or estimates are available.
4. Reproducible training configurations, variability estimates, and error analysis.
5. A documented comparison with existing OpenWhistle benchmark results that distinguishes this study's protocol from prior work.

## Dataset

The project uses the [OpenWhistle Classification Finetuning dataset](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Classification-Finetuning) and released OpenWhistle model resources.

- **Dataset:** `dolphinteam/OpenWhistle-Classification-Finetuning`
- **Hugging Face collection:** [NeurIPS 2026 OpenWhistle resources](https://huggingface.co/collections/dolphinteam/neurips26-openwhistle)
- **Task:** Ten-class dolphin whistle classification
- **Target:** The dataset's `label` field; the official label mapping will be verified from the dataset metadata.

Before training, the dataset audit will verify:

- Split sizes and class counts.
- Label names and class definitions.
- Audio duration, sampling rate, and file format.
- Missing or malformed examples.
- Recording/session identifiers and potential overlap between splits.
- Availability and reliability of SNR metadata or estimates.

Class imbalance is an important consideration. Aggregate accuracy alone may obscure weak performance on underrepresented classes, so macro-F1 and per-class metrics are included in the evaluation plan.

**Data-split policy:** Use the official training, validation, and test partitions. The validation set is for model selection; the test set is reserved for final evaluation. Any potential recording-session overlap will be audited and documented rather than assumed absent.

## Experimental Design

### Experiment A — ResNet-18 Baseline

**Purpose:** Establish a supervised baseline that learns from the labeled dataset without pretrained weights.

- **Input:** Log-Mel spectrograms computed from whistle clips.
- **Model:** ResNet-18 initialized from random weights, with a ten-class classification head.
- **Training:** Train on the official training split.
- **Model selection:** Select checkpoints using validation macro-F1.
- **Evaluation:** Report final test metrics using the selected checkpoint.

This experiment provides a reference point for evaluating the value of representations learned through pretraining.

### Experiment B — General Pretrained Audio Encoder

**Purpose:** Measure the value of representations learned from audio outside the target task.

Candidate encoders include [AVES-Bio](https://github.com/earthspecies/aves) or another compatible pretrained audio encoder. The final choice will depend on checkpoint availability, licensing, input requirements, and compute constraints.

Two strategies will be compared where feasible:

1. **Frozen encoder + linear head:** Keep the pretrained encoder frozen and train a classifier on its embeddings.
2. **End-to-end fine-tuning:** Train the classification head and update all or selected encoder layers.

The checkpoint, input representation, sampling rate, embedding dimension, and trainable layers will be recorded for each run. Model-specific preprocessing will follow the checkpoint's documented requirements.

### Experiment C — OpenWhistle Wav2Vec 2.0

**Purpose:** Evaluate representations pretrained specifically for the OpenWhistle domain.

- Start from a released OpenWhistle Wav2Vec 2.0 checkpoint.
- Verify the checkpoint's expected waveform format, sampling rate, and preprocessing.
- Train a classifier on frozen embeddings.
- Fine-tune the encoder if the available GPU budget permits.
- Compare results under the same data partitions and labeled-data budgets as the other models.

This is the central domain-specific comparison. The analysis will explicitly distinguish the current experimental protocol from results already reported in the OpenWhistle paper and related resources.

### Optional Experiment D — Fundamental-Frequency Contour Model

If time and reliable F0 tracks permit, an additional experiment will test whether explicit fundamental-frequency contours can provide a compact alternative to learned audio representations.

- **Input:** Supplied F0 tracks represented as sequences.
- **Candidate models:** A small 1D-CNN or bidirectional LSTM (BiLSTM).
- **Purpose:** Evaluate the information available in whistle-contour features.
- **Status:** Optional; it will not take priority over completing the three main experiments.

## Evaluation Protocol

### Primary Experiment: Full-Data Classification

Train each main model using the same official training split. Use the same validation and test partitions for all models, while allowing preprocessing to follow each model's documented input requirements.

Select checkpoints using **validation macro-F1**. After model selection, evaluate the selected checkpoint on the test set once for the primary final report.

Report:

| Metric | Purpose |
|---|---|
| Macro-F1 | Primary metric; gives equal weight to each class |
| Accuracy | Overall fraction of correct predictions |
| Balanced accuracy | Average recall across classes |
| Per-class precision, recall, and F1 | Identifies classes with weak performance |
| Confusion matrix | Shows systematic confusions between classes |
| Parameter count | Describes model size |
| Inference time | Helps compare practical deployment cost |

Inference timing will use a documented hardware and batch-size setup. Comparisons will account for whether preprocessing and encoder execution are included in the measured time.

## Label-Efficiency Study

To quantify how much labeled data each approach requires, train models using the following fractions of the official training set:

- **10%**
- **25%**
- **50%**
- **100%**

The validation and test partitions remain unchanged.

### Protocol

1. Sample stratified subsets from the training partition only, where the class counts permit it.
2. Use identical subset indices for each model at each label budget.
3. Use three random seeds for subset selection and training where compute permits.
4. Keep validation-based model selection consistent across models.
5. Report mean and standard deviation of test macro-F1 across seeds.
6. Plot test macro-F1 against the number or fraction of labeled training examples.

Very small budgets may not contain enough examples from every class for meaningful stratification. Any such constraints will be documented; examples will not be silently moved from validation or test into training.

## Robustness Analysis

The project will investigate whether performance changes with recording quality and whistle category.

Planned analyses include:

- Performance across SNR groups, if SNR values can be measured or reliably derived.
- Per-class precision, recall, and F1.
- Confusion matrices for each principal model.
- Review of representative false positives and false negatives.
- Comparison between frozen-feature and fine-tuned variants.
- Confidence intervals or other uncertainty estimates when the number of runs and evaluation examples support them.

SNR groups, filtering rules, and any derived quality measures will be defined before interpreting subgroup results. If reliable SNR estimates are unavailable, the limitation will be reported rather than replaced with unsupported assumptions.

## Implementation Roadmap

The project is planned over approximately six to eight weeks. The schedule may change based on checkpoint compatibility and compute availability.

| Week | Focus | Planned work |
|---|---|---|
| 1 | Setup and dataset audit | Review the OpenWhistle paper; verify splits, labels, metadata, and audio properties; inspect example spectrograms; establish experiment logs |
| 2 | Baseline | Build preprocessing and train ResNet-18; tune on validation data; save checkpoints and predictions |
| 3 | General transfer learning | Evaluate a frozen pretrained encoder; attempt fine-tuning; record memory use, training time, and convergence |
| 4 | In-domain transfer learning | Evaluate OpenWhistle Wav2Vec 2.0 embeddings; fine-tune if feasible |
| 5 | Label efficiency | Run 10%, 25%, 50%, and 100% label-budget experiments with multiple seeds where possible |
| 6 | Robustness and error analysis | Compare SNR groups and per-class errors; run the optional F0 model only if the primary experiments are stable |
| 7–8 | Writing and release | Prepare tables, learning curves, confusion matrices, limitations, and reproducibility instructions |

## Technology Stack

| Component | Planned tools |
|---|---|
| Development | Google Colab or Kaggle notebooks |
| Deep learning | PyTorch |
| Dataset loading | Hugging Face `datasets` |
| Audio processing | `torchaudio`, `librosa` |
| Pretrained models | Hugging Face Transformers and released project checkpoints |
| Metrics | `scikit-learn` |
| Experiment tracking | CSV/JSON logs or Weights & Biases |
| Visualization | Matplotlib, Seaborn |
| Version control | Git and GitHub |

### Compute Strategy

- Begin with a T4-class GPU when available.
- Use mixed-precision training where supported and numerically appropriate.
- Start with small batches and frozen encoders when GPU memory is constrained.
- Cache deterministic preprocessing and frozen embeddings where storage permits.
- Do not attempt to pretrain Wav2Vec 2.0 from scratch within this project timeline.

## Reproducibility

To make experiments easier to reproduce, each run should record:

- Model name, checkpoint identifier, and checkpoint revision where available.
- Dataset configuration and split identifiers.
- Random seeds and exact label-budget subset indices.
- Sampling rate, audio preprocessing, spectrogram parameters, and normalization.
- Optimizer, learning rate, batch size, epochs, scheduler, and early-stopping criteria.
- Frozen or trainable encoder layers.
- Validation-selected checkpoint and test predictions.
- Software versions, hardware, training duration, and inference-time settings.

Where licensing and dataset terms permit, the repository will include code, configuration files, metric summaries, and instructions to reproduce the experiments. Large audio files and model checkpoints may remain hosted at their original sources rather than being committed to Git.

## Repository Structure

The structure below is a proposed layout and may evolve during implementation.

```text
.
├── README.md
├── configs/
│   ├── resnet18.yaml
│   ├── general_encoder.yaml
│   └── openwhistle_wav2vec2.yaml
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_resnet18_baseline.ipynb
│   ├── 03_general_pretrained_encoder.ipynb
│   ├── 04_openwhistle_wav2vec2.ipynb
│   └── 05_label_efficiency_and_analysis.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── scripts/
├── results/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── LICENSE
```

## Results

**Results are not yet available.** This section will be updated after the experiments and evaluation checks are complete.

Planned reporting includes:

- Full-data comparison across the three main approaches.
- Macro-F1 learning curves for 10%, 25%, 50%, and 100% label budgets.
- Mean and standard deviation across random seeds.
- Per-class metrics and confusion matrices.
- SNR-stratified robustness analysis where supported by the data.
- Model size, inference time, and compute considerations.

No model is assumed to be best in advance. The conclusions will be based on the observed results and their limitations.

## Limitations and Scope

This study is constrained by available labeled data, GPU resources, and compatibility with released pretrained checkpoints. Results may depend on preprocessing choices, checkpoint provenance, class imbalance, and the composition of the official data splits.

The comparison is intended to quantify performance under a defined protocol, not to establish universal superiority of one model family. Existing OpenWhistle benchmark results will be acknowledged, and claims of novelty will be limited to the experimental questions and analyses actually conducted.

## References

- **OpenWhistle dataset:** [OpenWhistle Classification Finetuning on Hugging Face](https://huggingface.co/datasets/dolphinteam/OpenWhistle-Classification-Finetuning)
- **OpenWhistle model resources:** [NeurIPS 2026 OpenWhistle collection](https://huggingface.co/collections/dolphinteam/neurips26-openwhistle)
- **AVES:** [Earth Species Project AVES repository](https://github.com/earthspecies/aves)
- **Hugging Face Datasets:** [Documentation](https://huggingface.co/docs/datasets)
- **PyTorch:** [Documentation](https://pytorch.org/docs/stable/index.html)
- **scikit-learn:** [Documentation](https://scikit-learn.org/stable/)

Please consult the original OpenWhistle paper and each model checkpoint's documentation for exact methodology, citation, licensing, and input requirements.

---

*This repository documents an ongoing research project. Experimental settings, results, and conclusions will be updated as the work progresses.*
