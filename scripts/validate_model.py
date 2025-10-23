"""
Validate the fine-tuned medical NER model.
Tests the model on validation samples and compares outputs.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from typing import List, Dict

# ========================================
# CONFIGURATION
# ========================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "./final_model"  # or "your-username/llama3-medical-ner-lora"

# ========================================
# LOAD MODEL
# ========================================

print("="*80)
print("LOADING MODEL")
print("="*80)

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)
model.eval()

print(f"✓ Model loaded and ready for inference")

# ========================================
# INFERENCE FUNCTION
# ========================================

def generate_response(prompt_text: str, max_new_tokens: int = 512) -> str:
    """Generate a response for a given prompt."""
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical NER expert. Extract the requested entities from medical texts accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "assistant\n\n" in response:
        response = response.split("assistant\n\n")[-1]
    elif "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response.strip()

# ========================================
# LOAD TEST DATA (COMPLETELY UNSEEN)
# ========================================

print("\n" + "="*80)
print("LOADING TEST DATA")
print("="*80)

with open('../data/test.jsonl', 'r', encoding='utf-8') as f:
    test_samples = [json.loads(line) for line in f]

print(f"✓ Loaded {len(test_samples)} test samples")
print(f"⚠️  These samples were NOT used during training or validation")
print(f"⚠️  This is the FINAL evaluation on completely unseen data")

# ========================================
# TEST ON COMPLETELY UNSEEN EXAMPLES
# ========================================

print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)
print("This tests the model's ability to generalize to completely new data.")
print("Test set was never seen during training or validation monitoring.")
print("="*80)

# Test on first 5 examples
num_test_samples = 5

# Aggregate metrics
total_correct = 0
total_predicted = 0
total_expected = 0

for i, sample in enumerate(test_samples[:num_test_samples]):
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i+1}/{num_test_samples}")
    print(f"{'='*80}")
    
    # Show prompt (first 300 chars)
    print(f"\n📝 PROMPT:")
    print(f"{sample['prompt'][:300]}...")
    
    # Show expected output
    print(f"\n✓ EXPECTED OUTPUT:")
    print(f"{sample['completion']}")
    
    # Generate prediction
    print(f"\n🤖 MODEL OUTPUT:")
    prediction = generate_response(sample['prompt'])
    print(f"{prediction}")
    
    # Calculate metrics
    print(f"\n📊 EVALUATION:")
    expected_items = set([item.strip() for item in sample['completion'].split('\n') if item.strip()])
    predicted_items = set([item.strip() for item in prediction.split('\n') if item.strip()])
    
    common = expected_items & predicted_items
    missing = expected_items - predicted_items
    extra = predicted_items - expected_items
    
    # Update aggregate counts
    total_correct += len(common)
    total_predicted += len(predicted_items)
    total_expected += len(expected_items)
    
    # Per-sample metrics
    accuracy = len(common) / len(expected_items) * 100 if len(expected_items) > 0 else 0
    precision = len(common) / len(predicted_items) * 100 if len(predicted_items) > 0 else 0
    recall = len(common) / len(expected_items) * 100 if len(expected_items) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if expected_items == predicted_items:
        print("✅ EXACT MATCH!")
    else:
        print(f"  ✓ Correct: {len(common)}/{len(expected_items)}")
        print(f"  ✗ Missed: {len(missing)}")
        print(f"  ⚠ Extra: {len(extra)}")
        print(f"\n  📈 Metrics:")
        print(f"    Accuracy:  {accuracy:.1f}%")
        print(f"    Precision: {precision:.1f}%")
        print(f"    Recall:    {recall:.1f}%")
        print(f"    F1 Score:  {f1:.1f}%")
        
        if missing:
            print(f"\n  Missing from prediction:")
            for item in list(missing)[:3]:
                if item.strip():
                    print(f"    {item}")
        
        if extra:
            print(f"\n  Extra in prediction:")
            for item in list(extra)[:3]:
                if item.strip():
                    print(f"    {item}")

# ========================================
# AGGREGATE METRICS
# ========================================

print("\n" + "="*80)
print("AGGREGATE METRICS ACROSS TEST SAMPLES")
print("="*80)

# Calculate aggregate metrics
aggregate_precision = total_correct / total_predicted * 100 if total_predicted > 0 else 0
aggregate_recall = total_correct / total_expected * 100 if total_expected > 0 else 0
aggregate_f1 = 2 * (aggregate_precision * aggregate_recall) / (aggregate_precision + aggregate_recall) if (aggregate_precision + aggregate_recall) > 0 else 0
aggregate_accuracy = total_correct / total_expected * 100 if total_expected > 0 else 0

print(f"\nEvaluated on {num_test_samples} test samples:")
print(f"\n📊 Overall Performance:")
print(f"  Total expected entities:  {total_expected}")
print(f"  Total predicted entities: {total_predicted}")
print(f"  Correctly predicted:      {total_correct}")
print(f"\n📈 Aggregate Metrics:")
print(f"  Accuracy:  {aggregate_accuracy:.2f}%")
print(f"  Precision: {aggregate_precision:.2f}% (fewer false positives)")
print(f"  Recall:    {aggregate_recall:.2f}% (fewer false negatives)")
print(f"  F1 Score:  {aggregate_f1:.2f}% (balanced metric)")

print(f"\n💡 Interpretation:")
print(f"  - Precision: Of all entities predicted, {aggregate_precision:.1f}% were correct")
print(f"  - Recall: Of all actual entities, {aggregate_recall:.1f}% were found")
print(f"  - F1: Harmonic mean balancing precision and recall")

# ========================================
# CUSTOM TEST CASES
# ========================================

print("\n" + "="*80)
print("CUSTOM TEST CASES")
print("="*80)

custom_tests = [
    {
        "task": "Chemical Extraction",
        "prompt": """The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the chemicals mentioned.

A patient was treated with aspirin and ibuprofen for pain relief. The combination of these NSAIDs proved effective in reducing inflammation.

List of extracted chemicals:
"""
    },
    {
        "task": "Disease Extraction",
        "prompt": """The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the diseases mentioned.

The patient presented with hypertension, diabetes mellitus, and chronic kidney disease. Laboratory findings revealed proteinuria and elevated creatinine levels.

List of extracted diseases:
"""
    },
]

for i, test in enumerate(custom_tests, 1):
    print(f"\n--- Custom Test {i}: {test['task']} ---")
    print(f"\nPrompt:\n{test['prompt']}")
    print(f"\nModel Output:")
    result = generate_response(test['prompt'])
    print(result)

# ========================================
# SUMMARY
# ========================================

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print(f"\nTested on {num_test_samples} validation samples")
print(f"Model performed inference successfully")
print(f"\nFor comprehensive evaluation, consider:")
print(f"  - Computing F1 scores for entity extraction")
print(f"  - Evaluating on full validation set")
print(f"  - Testing on completely unseen medical texts")
print(f"  - Comparing with base model (ablation study)")
