# Medical IE Notebook Fix Pack — Data Split, Precision, and Task Output
_Date: 2025-11-08_

This guide is a drop‑in set of changes to fix your dataset split leakage, reduce false positives, and standardize outputs — so another LLM can safely patch your notebooks.

---

## TL;DR (what to change)

1. **Deduplicate before splitting** (by normalized prompt) and then do the stratified 80/10/10 split. Add hard asserts that _no_ prompts overlap across splits.
2. **Make evaluation deterministic** (no sampling) and add **post‑filters** that keep items only if they appear in the source text (chemicals/diseases).  
3. Redesign **influences** to output `chemical | disease` pairs and only accept pairs where **both sides** appear in the text (optionally with proximity).
4. **Tighten training prompts** and lower LR to **5e‑5**; add hard negatives if precision is still low.

---

## 0) Reusable utilities (paste near the top of the notebook)

```python
# ===== Utilities: normalization, hashing, parsing =====
import re, json, hashlib
from collections import Counter

def dehyphenate(s: str) -> str:
    # Join words broken across lines with hyphens + whitespace
    return re.sub(r"(\w+)-\s+(\w+)", r"\1\2", s)

def normalize_text(s: str) -> str:
    s = dehyphenate(s or "")
    s = s.lower()
    s = re.sub(r"[\u00A0\t\r\n]+", " ", s)     # spaces/newlines
    s = re.sub(r"\s+", " ", s).strip()
    return s

def prompt_hash(prompt: str) -> str:
    return hashlib.md5(normalize_text(prompt).encode("utf-8")).hexdigest()

def parse_bullets(text: str):
    items = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*[-*]\s*(.+?)\s*$", line)
        if m:
            items.append(m.group(1))
    return items

def normalize_item(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\-\s]+", " ", s)
    s = re.sub(r"[\.,;:]+$", "", s).strip()
    return s

def in_text(item: str, text: str) -> bool:
    return normalize_item(item) in normalize_text(text)

def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out
```

---

## 1) Data clean‑up and **leak‑free** stratified split

> Replace your current split cell with this. It deduplicates by normalized prompt hash, computes task labels from your prompt template, splits 80/10/10 with stratify, writes files, and **asserts** no cross‑split overlap.

```python
# ===== 1) Deduplicate + Stratified Split (80/10/10) =====
from sklearn.model_selection import train_test_split
import random, os, time

SPLIT_SEED = 42  # set once, keep stable
random.seed(SPLIT_SEED)

# 1.1 Load your full dataset into `data` (list of dicts with 'prompt','completion')
# Example: data = [json.loads(line) for line in open('both_rel_instruct_all.jsonl','r',encoding='utf-8')]

assert isinstance(data, list) and len(data) > 0, "Load `data` first"

# 1.2 Deduplicate by normalized prompt
seen = set()
clean = []
for row in data:
    ph = prompt_hash(row["prompt"])
    if ph in seen: 
        continue
    seen.add(ph)
    # Optional: also clean completion lines (dedupe case variants)
    items = unique_preserve_order(parse_bullets(row.get("completion","")))
    items_norm = unique_preserve_order([normalize_item(x) for x in items if x.strip()])
    row["completion"] = "\n".join(f"- {x}" for x in items_norm)
    clean.append(row)

data = clean
print(f"After dedup: {len(data)} rows")

# 1.3 Task labels from your prompt template
def task_from_prompt(prompt: str) -> str:
    p = normalize_text(prompt)
    if "list of extracted chemicals" in p: return "chemicals"
    if "list of extracted diseases"  in p: return "diseases"
    if "list of extracted influences" in p: return "influences"
    return "other"

labels = [task_from_prompt(r["prompt"]) for r in data]
print("Label distribution:", Counter(labels))

# 1.4 First split: 80% train, 20% temp
train_data, temp_data, train_y, temp_y = train_test_split(
    data, labels, test_size=0.2, random_state=SPLIT_SEED, stratify=labels
)

# 1.5 Second split: 10% val, 10% test (split temp 50/50)
val_data, test_data, val_y, test_y = train_test_split(
    temp_data, temp_y, test_size=0.5, random_state=SPLIT_SEED+1, stratify=temp_y
)

def check_leak(a, b, name):
    ha = {prompt_hash(r["prompt"]) for r in a}
    hb = {prompt_hash(r["prompt"]) for r in b}
    overlap = ha & hb
    print(f"{name}: overlap={{len(overlap)}}")
    assert len(overlap) == 0, f"Leak detected between {{name}}"

check_leak(train_data, val_data, "train ∩ val")
check_leak(train_data, test_data, "train ∩ test")
check_leak(val_data,   test_data, "val ∩ test")

# 1.6 Write files
out_train = "train.jsonl"
out_val   = "validation.jsonl"
out_test  = "test.jsonl"

for path, split in [(out_train,train_data),(out_val,val_data),(out_test,test_data)]:
    with open(path,"w",encoding="utf-8") as f:
        for r in split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Wrote:", out_train, out_val, out_test)
```

---

## 2) Deterministic generation for **evaluation**

> Turn **off** sampling. Greedy decoding prevents “creative” additions that tank precision.

```python
# ===== 2) Deterministic generation =====
def generate_list(model, tokenizer, prompt, max_new_tokens=128):
    from torch import no_grad
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text
```

---

## 3) Post‑filters to raise precision

### 3.1 Chemicals / Diseases (span‑based)
Keep only items that appear in the source text (after normalization). Deduplicate.

```python
def filter_items_against_text(pred_items, prompt_text):
    keep = []
    for it in pred_items:
        if in_text(it, prompt_text):
            keep.append(normalize_item(it))
    return unique_preserve_order(keep)

def extract_list_from_generation(gen_text):
    # Parse bullets from the model output
    return parse_bullets(gen_text)
```

### 3.2 Influences as pairs (preferred spec)
Change output to **`chemical | disease`** per line and keep the pair only if **both tokens** appear in the prompt.

```python
def parse_pairs(gen_text):
    pairs = []
    for line in parse_bullets(gen_text):
        parts = [p.strip() for p in line.split("|")]
        if len(parts)==2:
            pairs.append(tuple(parts))
    return unique_preserve_order(pairs)

def filter_pairs_against_text(pairs, prompt_text):
    kept = []
    for chem, dis in pairs:
        if in_text(chem, prompt_text) and in_text(dis, prompt_text):
            kept.append((normalize_item(chem), normalize_item(dis)))
    # Deduplicate normalized pairs
    seen=set(); out=[]
    for p in kept:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

# Temporary fallback if you still have sentence outputs
def sentence_to_pair(line):
    # e.g., "Chemical X influences disease Y"
    m = re.match(r"^\s*chemical\s+(.+?)\s+influences\s+disease\s+(.+?)\s*$", line, re.I)
    return (m.group(1), m.group(2)) if m else None
```

---

## 4) Evaluation loop (per‑task metrics + sanity prints)

```python
# ===== 4) Evaluation =====
from statistics import mean

def f1(p, r): 
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)

def eval_split(model, tokenizer, jsonl_path, max_new_tokens=128):
    gold_total = {"chemicals":0,"diseases":0,"influences":0}
    pred_total = {"chemicals":0,"diseases":0,"influences":0}
    tp_total   = {"chemicals":0,"diseases":0,"influences":0}

    examples_fp = []
    examples_fn = []

    with open(jsonl_path,"r",encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt = row["prompt"]
            gold_items = [normalize_item(x) for x in parse_bullets(row.get("completion",""))]
            task = task_from_prompt(prompt)

            # generate
            gen = generate_list(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            pred_raw = extract_list_from_generation(gen)

            if task in {"chemicals","diseases"}:
                pred = filter_items_against_text(pred_raw, prompt)
            elif task == "influences":
                pairs = parse_pairs(gen)  # expecting new spec
                pred = [f"{c} | {d}" for (c,d) in filter_pairs_against_text(pairs, prompt)]
                gold_items = [normalize_item(x) for x in gold_items]  # keep as strings for now
            else:
                pred = []

            # sets for metrics
            gs = set(gold_items)
            ps = set(pred)

            tp = len(gs & ps)
            fp = len(ps - gs)
            fn = len(gs - ps)

            gold_total[task] += len(gs)
            pred_total[task] += len(ps)
            tp_total[task]   += tp

            if fp and len(examples_fp) < 8:
                examples_fp.append({"task":task,"prompt_preview":prompt[:120]+"...","pred_extras":list(ps-gs)[:5]})
            if fn and len(examples_fn) < 8:
                examples_fn.append({"task":task,"prompt_preview":prompt[:120]+"...","missed":list(gs-ps)[:5]})

    # Metrics
    for t in ["chemicals","diseases","influences"]:
        P = 0.0 if pred_total[t]==0 else tp_total[t]/pred_total[t]
        R = 0.0 if gold_total[t]==0 else tp_total[t]/gold_total[t]
        print(f"{t:11s}  P: {P*100:5.1f}%  R: {R*100:5.1f}%  F1: {f1(P,R)*100:5.1f}%  "
              f"(TP={tp_total[t]}  Pred={pred_total[t]}  Gold={gold_total[t]})")

    if examples_fp:
        print("\nExamples of false positives:")
        for e in examples_fp: print(e)
    if examples_fn:
        print("\nExamples of false negatives:")
        for e in examples_fn: print(e)
```

---

## 5) Training‑time guardrails (recommended)

```python
# ===== 5) Training argument tweaks =====
LEARNING_RATE = 5e-5      # was 2e-4, reduce to curb over‑prediction
NUM_EPOCHS    = 3         # OK to keep; if underfitting, 4–5
BATCH_SIZE    = 4         # keep as per VRAM
GRADIENT_ACCUMULATION = 4 # adjust if needed

# Tighten the instruction template:
TRAINING_INSTRUCTION = (
    "Return ONLY entities that appear verbatim in the article.\n"
    "Output one item per line, each starting with '- '.\n"
    "If none exist, return nothing.\n"
    "Do not add explanations or examples."
)
```

Add **hard negatives**: passages with many entity‑looking words but few/no valid items. They teach the model to avoid guessing.

---

## 6) Optional: synonym mapping (evaluation only)

```python
# synonyms.csv -> two columns: variant, canonical
def load_synonyms(path="synonyms.csv"):
    table = {}
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if "," in line:
                    a,b = line.strip().split(",",1)
                    table[normalize_item(a)] = normalize_item(b)
    except FileNotFoundError:
        pass
    return table

SYN = load_synonyms()

def canonicalize(s):
    s = normalize_item(s)
    return SYN.get(s, s)
```

Map gold and predictions through `canonicalize` before set comparisons if you maintain a synonyms list.

---

## 7) Post‑split leakage check (standalone)

```python
# ===== 7) Sanity check: zero overlap across written files =====
def hash_file(path):
    out=set()
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            out.add(prompt_hash(r["prompt"]))
    return out

Htrain = hash_file("train.jsonl")
Hval   = hash_file("validation.jsonl")
Htest  = hash_file("test.jsonl")

print("train∩val:", len(Htrain & Hval))
print("train∩test:", len(Htrain & Htest))
print("val∩test:", len(Hval & Htest))

assert not (Htrain & Hval),   "Leak train∩val"
assert not (Htrain & Htest),  "Leak train∩test"
assert not (Hval & Htest),    "Leak val∩test"
print("✓ No cross‑split leakage")
```

---

## Run order

1) Load full dataset into `data`.  
2) Section **1** (Dedup + Split).  
3) Load model/tokenizer.  
4) Section **2** (Deterministic generation).  
5) Section **3** (Post‑filters).  
6) Section **4** (Evaluation).  
7) Section **5** (Training tweaks) as needed; re‑fine‑tune; re‑evaluate.  
8) Section **6** optionally.
