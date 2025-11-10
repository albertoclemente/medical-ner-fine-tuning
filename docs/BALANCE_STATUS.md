# Dataset Balance Status Report

## ✅ FIXED: Data Files are Now Properly Balanced

### Current State (as of Nov 4, 2025)

#### `/data/` Directory (UPDATED - Use These!)
- ✅ **train.jsonl**: 2,400 examples (33.3% chemical, 66.7% disease)
- ✅ **validation.jsonl**: 300 examples (33.3% chemical, 66.7% disease)  
- ✅ **test.jsonl**: 300 examples (33.3% chemical, 66.7% disease)
- **Status**: Properly stratified, ready to use

#### `/notebooks/` Directory (OLD - Will be overwritten)
- ❌ **train.jsonl**: 2,400 examples (41.7% chemical, 58.3% disease)
- ❌ **validation.jsonl**: 300 examples (0% chemical, 100% disease) ⚠️
- ❌ **test.jsonl**: 300 examples (0% chemical, 100% disease) ⚠️
- **Status**: OLD imbalanced splits from Oct 29
- **Note**: These will be recreated when you run the notebook

## What Happens When You Run the Notebook

The notebook (`Medical_NER_Fine_Tuning.ipynb`):
1. Loads `both_rel_instruct_all.jsonl` from notebooks directory
2. Creates NEW stratified splits using `train_test_split` with `stratify=` parameter
3. Saves to `notebooks/*.jsonl` (overwrites old files)
4. Uses the newly created in-memory splits for training

✅ **The notebook code is CORRECT** - it will create balanced splits!

## Recommended Actions

1. **Option A: Use the notebook as-is**
   - Just run the notebook - it will create proper stratified splits
   - The old notebooks/*.jsonl files will be overwritten with good ones
   
2. **Option B: Use pre-split files from /data/**
   - Modify notebook to load from `/data/train.jsonl` instead of splitting
   - Faster startup, consistent splits across runs
   - Good for reproducibility

3. **Clean up documentation**
   - Remove references to "relationship extraction" (dataset only has chemical + disease)
   - Update percentages from "33.3% each" to "33.3% chemical, 66.7% disease"

## Issues Found & Fixed

### Before:
- `/data/` splits were created with `scripts/split_data.py` WITHOUT stratification
- Validation/test had 0% chemical extraction examples
- Model would only be evaluated on disease extraction during training

### After:
- Created `scripts/split_data_stratified.py` with proper stratification
- All splits now have proportional task distribution
- Both chemical and disease extraction will be monitored during training

## Next Steps

✅ Data is ready for training!
✅ Notebook code is correct (uses stratification)
✅ Just run the notebook and it will work properly

The only thing to fix: Update comments/docs to reflect actual t   - Fasterution (2 tasks, not 3).
