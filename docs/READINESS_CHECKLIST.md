# Training & Evaluation Readiness Checklist

## Status: ✅ READY FOR NEW TRAINING

Both notebooks have been fixed and are ready for training with properly balanced data.

---

## Training Notebook: `Medical_NER_Fine_Tuning.ipynb`

### ✅ Fixed Issues:
- [x] Stratified splitting implemented (`stratify=labels`)
- [x] Guarantees EXACT 33.3% of each task type in all splits
- [x] Verification cell added to confirm balanced distribution
- [x] Data integrity check (no leakage between splits)
- [x] Documentation updated with warnings about shuffle importance

### 📋 Pre-Training Checklist:
- [ ] Update `HF_TOKEN` in Section 0 (cell 3)
- [ ] Update `WANDB_API_KEY` in Section 0 (cell 3) - optional
- [ ] Update `HF_USERNAME` in Section 3 (cell 9)
- [ ] Ensure source data file exists: `both_rel_instruct_all.jsonl`
- [ ] Run Section 5 to load and explore data
- [ ] Run Section 6 to create stratified splits (CRITICAL!)
- [ ] Verify split distribution shows ✅ markers (exactly 33.3% each)
- [ ] Continue with training sections 7-13

### 📊 Expected Split Distribution (After Section 6):
```
Training (2,400 samples):
  ✅ Chemical Extraction: 800 (33.3%)
  ✅ Disease Extraction: 800 (33.3%)
  ✅ Relationship Extraction: 800 (33.3%)

Validation (300 samples):
  ✅ Chemical Extraction: 100 (33.3%)
  ✅ Disease Extraction: 100 (33.3%)
  ✅ Relationship Extraction: 100 (33.3%)

Test (300 samples):
  ✅ Chemical Extraction: 100 (33.3%)
  ✅ Disease Extraction: 100 (33.3%)
  ✅ Relationship Extraction: 100 (33.3%)
```

### ⏱️ Training Time Estimate:
- **GPU**: A100 (40GB)
- **Expected Duration**: 2-3 hours
- **Epochs**: 3
- **Checkpoints**: Saved every 100 steps to HuggingFace Hub

---

## Evaluation Notebook: `Medical_NER_Evaluation.ipynb`

### ✅ Fixed Issues:
- [x] Header updated with new training instructions
- [x] Test data path corrected to `notebooks/test.jsonl`
- [x] Case-insensitive evaluation implemented
- [x] False positive analysis section added
- [x] Balanced test set expected after new splits

### 📋 Pre-Evaluation Checklist:
- [ ] Complete training in `Medical_NER_Fine_Tuning.ipynb`
- [ ] Update `HF_MODEL_ID` in Section 3 (cell 9) with NEW model ID
- [ ] Update `HF_TOKEN` in Section 0 (cell 3)
- [ ] Verify test.jsonl exists and is balanced (33.3% each task)
- [ ] Run evaluation on full test set (300 samples)

### 📊 Expected Test Results (With Balanced Data):
```
Aggregate Metrics:
  - Precision: Should improve significantly on relationship extraction
  - Recall: More balanced across all three task types
  - F1 Score: Higher overall due to balanced training
  - False Positive Rate: Should decrease (model not guessing as much)
  - False Negative Rate: Should decrease (model saw all tasks equally)
```

---

## Verification Commands

Run these to verify everything is ready:

### 1. Check Training Notebook Has Stratified Splitting:
```bash
grep -n "stratify=" Medical_NER_Fine_Tuning.ipynb | head -5
```

Expected output: Should show `stratify=stratify_labels` in splitting code

### 2. Verify Current Split Files Need Regeneration:
```bash
python3 verify_data_splits.py
```

Expected output: Should show imbalanced distribution (100% relationship in test/val)

### 3. Check Source Data Exists:
```bash
# Training notebook expects this file
ls -lh both_rel_instruct_all.jsonl 2>/dev/null || echo "⚠️ Source file not found - check location"
```

---

## Step-by-Step Workflow

### Phase 1: Prepare Data (5 minutes)
1. Open `Medical_NER_Fine_Tuning.ipynb`
2. Update credentials (HF_TOKEN, WANDB_API_KEY, HF_USERNAME)
3. Run Sections 0-5 (setup, imports, config, data loading)
4. **CRITICAL**: Run Section 6 to create stratified splits
5. Verify ✅ markers showing exact 33.3% distribution
6. Save generated files: train.jsonl, validation.jsonl, test.jsonl

### Phase 2: Train Model (2-3 hours)
7. Run Sections 7-12 (data formatting, model loading, LoRA config, training)
8. Monitor training in Weights & Biases dashboard
9. Wait for training completion
10. Note the HuggingFace Hub model ID (e.g., `albyos/llama3-medical-ner-lora-20251031_120000`)

### Phase 3: Evaluate Model (10-15 minutes)
11. Open `Medical_NER_Evaluation.ipynb`
12. Update `HF_MODEL_ID` with new trained model
13. Update `HF_TOKEN` 
14. Run all cells sequentially
15. Review metrics and compare with old (imbalanced) model

---

## Key Files

### Generated During Training:
- `train.jsonl` - 2,400 samples (80%)
- `validation.jsonl` - 300 samples (10%)
- `test.jsonl` - 300 samples (10%)
- `final_model/` - Local model checkpoint
- HuggingFace Hub model - Online checkpoint

### Required Before Training:
- `both_rel_instruct_all.jsonl` - Source data (3,000 samples)

### Verification Scripts:
- `verify_data_splits.py` - Check if splits are balanced
- `READINESS_CHECKLIST.md` - This file

---

## Troubleshooting

### Issue: "Source file not found"
**Solution**: Ensure `both_rel_instruct_all.jsonl` is in the notebooks directory or update path in Section 5

### Issue: Split verification shows imbalance
**Solution**: Re-run Section 6 of training notebook - stratified splitting should fix this

### Issue: Old test.jsonl has 100% relationships
**Solution**: This is expected! Re-run Section 6 to generate new balanced splits

### Issue: Model ID not found in evaluation
**Solution**: Update `HF_MODEL_ID` in Medical_NER_Evaluation.ipynb with your NEW model ID

---

## Success Criteria

✅ Training notebook generates perfectly balanced splits (33.3% each)
✅ Training completes without errors
✅ Model uploaded to HuggingFace Hub
✅ Evaluation shows improved metrics on relationship extraction
✅ False positive/negative rates are balanced across all task types

---

## Questions or Issues?

Review the notebooks for inline documentation and warnings.
