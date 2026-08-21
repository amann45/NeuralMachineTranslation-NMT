# Comparative Analysis of Neural Machine Translation Architectures

## 📌 Project Overview

This project presents a comparative experimental study of four Neural Machine Translation (NMT) architectures for **English → Korean** translation:

1. **Vanilla RNN**
2. **GRU Encoder–Decoder (Seq2Seq)**
3. **Encoder–Decoder with Bahdanau Additive Attention**
4. **Encoder–Decoder with Luong Multiplicative Attention**

The project follows the experimental requirements of the assignment by using a bilingual dataset from the **ManyThings.org Anki Language Datasets** repository and evaluating the models under comparable training and testing conditions.

The main goal is to investigate how progressively stronger sequence-modeling and attention mechanisms affect translation quality, language-modeling performance, computational cost, and inference latency.

---

## 🎯 Objectives

The project aims to:

- Implement four NMT architectures from scratch using PyTorch.
- Use a bilingual dataset from the ManyThings.org Anki Language Datasets.
- Evaluate models using quantitative and qualitative criteria.
- Compare translation quality using **BLEU**.
- Measure **test cross-entropy loss** and **perplexity**.
- Measure **inference latency**.
- Compare the number of trainable parameters and training time.
- Analyze the strengths and limitations of attention-based and non-attention-based architectures.
- Provide reproducible code, saved results, plots, and model checkpoints.

---

## 📚 Dataset

The experiment uses the **English–Korean** language pair from the ManyThings.org Anki Language Datasets:

**Dataset:** `kor-eng.zip`  
**Source:** [ManyThings.org Anki Language Datasets](https://www.manythings.org/anki/)

The notebook automatically downloads the dataset and extracts the translation file.

### Dataset statistics

| Item | Value |
|---|---:|
| Raw usable sentence pairs | 6,394 |
| After sequence-length filtering | 6,392 |
| Training set | 5,113 |
| Validation set | 639 |
| Test set | 640 |
| Source vocabulary | 2,044 |
| Target vocabulary | 2,363 |

The experiment uses all 6,392 filtered sentence pairs because the configured maximum of 50,000 pairs is larger than the available dataset.

---
## ⚙️ Experimental Configuration

The models use the following common settings:

| Configuration | Value |
|---|---:|
| Embedding dimension | 256 |
| Hidden dimension | 256 |
| Maximum sequence length | 30 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Cross-Entropy Loss |
| Gradient clipping | 1.0 |
| Epochs | 10 |
| Initial teacher forcing | 1.0 |
| Final teacher forcing | 0.5 |
| Train split | 80% |
| Validation split | 10% |
| Test split | 10% |
| Device used in recorded run | CPU |

---
## Reproducibility

The notebook explicitly uses deterministic seeding across Python, NumPy, and PyTorch.

**Important assignment requirement:**

```text
random_state = <your_roll_number>
```

ROLL NO: ACE080BCT010

```python
SEED = 10
```
---
## 📊 Evaluation Metrics

The models are evaluated using multiple complementary criteria.

### BLEU

Corpus BLEU is used as the primary automatic translation-quality metric.

The implementation also calculates mean sentence-level BLEU.

### Perplexity

Perplexity is calculated from test cross-entropy loss:

```text
PPL = exp(test_loss)
```

Lower perplexity indicates better predictive performance on the target sequence.

### Inference latency

The experiment measures:

- Mean latency
- Median latency
- 95th-percentile latency (P95)

Latency is reported in milliseconds per sentence.

### Model size

The number of trainable parameters is recorded for each model.

### Training time

Total training time is recorded in seconds and reported in minutes.

### Qualitative evaluation

Representative examples are selected from:

- Simple sentences: ≤ 5 source tokens
- Complex sentences: 6–12 source tokens
- Long sentences: > 12 source tokens

The predicted translations are compared against the reference Korean translations.

---

## 📈 Experimental Results

**These values should be regenerated after changing the seed to the required roll number.**

| Model | Parameters | BLEU-4 | Sentence BLEU | Perplexity | Mean Latency (ms) | P95 Latency (ms) | Training Time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla RNN | 1,998,651 | ~0.000 | 0.276 | 213.49 | 13.59 | 16.35 | 11.99 |
| GRU Seq2Seq | 2,524,987 | 0.196 | 0.595 | 100.30 | 14.47 | 16.61 | 9.66 |
| Bahdanau Attention | 4,062,779 | 0.752 | 1.163 | 49.60 | 38.74 | 45.89 | 21.94 |
| Luong Attention | 3,996,987 | 0.476 | 0.902 | 59.11 | 27.61 | 31.02 | 20.63 |

> **Note:** The Vanilla RNN corpus BLEU is effectively zero in the recorded run because the generated hypotheses showed extremely poor higher-order n-gram overlap with the references. This is consistent with the qualitative examples, where the model repeatedly generated unrelated phrases and tokens.

---
## 📁 Repository Structure

A recommended GitHub repository structure is:

```text
nmt-comparative-analysis/
│
├── README.md
├── COMPARATIVE_ANALYSIS_OF_NMT_models_architectures.ipynb
│
├── nmt_comparative_analysis/
│   ├── data/
│   │   └── README.md
│   │
│   ├── checkpoints/
│   │   ├── vanilla_rnn.pt
│   │   ├── gru_seq2seq.pt
│   │   ├── bahdanau_attention.pt
│   │   └── luong_attention.pt
│   │
│   ├── plots/
│   │   ├── test_bleu_comparison.png
│   │   ├── inference_latency_comparison.png
│   │   └── bleu_vs_latency_tradeoff.png
│   │
│   └── results/
│       ├── quantitative_results.csv
│       └── qualitative_translations.csv
│
└── report/
    └── Report.pdf
```

The exact contents may vary depending on which generated artifacts are included in the final GitHub repository.

---
