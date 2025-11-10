# Deep Data Exploration Summary

**Date**: November 10, 2025  
**Notebook**: `notebooks/analysis/Data_Exploration_Deep_Dive.ipynb`

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total samples** | 3,000 |
| **Task distribution** | Chemical Extraction: 1,000<br>Disease Extraction: 1,000<br>Relationship Extraction: 1,000 |
| **Balance** | ✓ Well-balanced |

---

## Prompt Characteristics

| Metric | Value |
|--------|-------|
| **Median length** | 1,357 characters |
| **Median words** | 195 words |
| **Range** | 345-4,018 characters |
| **Recommended max_length** | 2048 tokens |

---

## Completion Patterns

| Metric | Value |
|--------|-------|
| **Median items per sample** | 3 |
| **Total unique chemicals** | 1,578 |
| **Total unique diseases** | 2,199 |
| **Relationship format** | sentence: 2,050 |

---

## Entity Naming Patterns

| Metric | Value |
|--------|-------|
| **Avg chemical name** | 11.1 chars, 1.2 words |
| **Avg disease name** | 14.9 chars, 1.7 words |
| **Hyphenated entities** | ~459 |
| **Special characters found** | 13 |

---

## Data Quality

| Metric | Value |
|--------|-------|
| **Empty completions** | 0 (0.00%) |
| **Duplicate prompts** | 0 (0.00%) |
| **Quality score** | ✓ High |

---

## Vocabulary

| Metric | Value |
|--------|-------|
| **Total vocabulary** | 13,710 unique words |
| **Active vocabulary** (>1 occurrence) | 13,678 |
| **Rare words** (singleton) | 32 |

---

## Key Recommendations for Fine-Tuning

### ✓ Data Split

- Use **80/10/10** train/val/test split
- **CRITICAL**: Apply stratified splitting (maintain task distribution)
- **CRITICAL**: Deduplicate prompts before splitting
- Expected sizes: ~2,400 train, ~300 val, ~300 test

### ✓ Format Standardization

- ⚠️ **CONVERT relationships from OLD to NEW format during preprocessing**
  - OLD: `'chemical X influences disease Y'`
  - NEW: `'X | Y'`
- Preserve hyphens in entity names (e.g., `'type-2 diabetes'`)
- System prompt should specify: `'- chemical | disease'` format

### ✓ Tokenization

- Recommended max_length: **2048 tokens**
- Padding side: **right**
- Truncation: **enabled**

### ✓ Training Parameters

- Batch size: **4-8 per device**
- Gradient accumulation: **4** (effective batch = 16-32)
- Learning rate: **5e-5** (conservative for LoRA)
- Epochs: **5** (sufficient for format learning)
- Scheduler: **cosine decay with warmup**

### ✓ Evaluation Strategy

- Metric: **F1 score per task type**
- Normalize entities before comparison (lowercase, trim)
- Use **word boundaries** for matching (prevent substring false positives)
- Track chemical F1, disease F1, influence F1 separately

---

## ✅ Analysis Complete

**Status**: Ready for fine-tuning with all recommendations applied.

**Next Steps**:
1. Run `Medical_NER_Fine_Tuning_new_20251108.ipynb` with format conversion
2. Monitor training with W&B
3. Evaluate with `Medical_NER_Evaluation_run03_20251108.ipynb`

---

**Generated from**: Data Exploration Deep Dive Notebook  
**Analysis Date**: November 10, 2025
