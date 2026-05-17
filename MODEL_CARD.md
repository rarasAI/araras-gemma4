---
base_model: unsloth/gemma-4-E4B-it
library_name: peft
pipeline_tag: text-generation
license: gemma
language:
- pt
- en
tags:
- gemma
- gemma4
- rare-disease
- brazilian-portuguese
- portuguese
- clinical
- medical
- sus
- ceaf
- pcdt
- conitec
- hpo
- orpha
- omim
- lora
- sft
- qlora
- unsloth
- trl
- edge
- mobile
- llama-cpp
- gguf
datasets:
- Raras-AI/araras-rare-disease-pt-v4
metrics:
- accuracy
- recall
model-index:
- name: araras-gemma4-e4b-v4-sota
  results:
  - task:
      type: text-generation
      name: Rare disease diagnostic differential (PT-BR)
    dataset:
      type: Raras-AI/rarebench-br
      name: RareBench-BR L5_realsus (DataSUS APAC-anchored)
    metrics:
    - type: recall@1_strict_orpha_with_pipeline
      value: 0.875
      name: R@1 strict (with canonical ORPHA lookup post-processor)
    - type: recall@3_strict_orpha_with_pipeline
      value: 0.875
      name: R@3 strict
    - type: pcdt_correct_rate
      value: 1.0
      name: Track B PCDT-correct (recommended CEAF drug matches real SUS dispensation, 22/22)
---

# Araras-Gemma4-E4B — open Gemma 4 fine-tune for Brazilian rare-disease care

**The first Gemma 4 fine-tune for rare diseases in Brazilian Portuguese, grounded in real SUS dispensation data.**

> *I'm a rare disease patient. It took me twenty years to get my diagnosis. Araras is what I wish someone had handed me in 2006.* — Dimas, founder, paciente raro

Submission to the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon).

## Model

- **Base**: `unsloth/gemma-4-E4B-it` (8B total / 4.5B effective parameters)
- **Method**: QLoRA SFT with Unsloth — `r=8, α=8, dropout=0`, NEFTune α=5, `train_on_responses_only`
- **Training data**: 120,740 train / 5,137 val examples (deduplicated), all PT-BR rare-disease content:
  - 108k curated rare-disease Q&A (HPO/OMIM/ORPHA-grounded)
  - 26k MedPT PT-BR clinical pairs
  - 10k ultra-rare disease longitudinal cases
  - 10k tool-calling examples (HPO normalize, PCDT lookup, CID-10 map)
  - 5.7k knowledge-graph triples (RarasNet Neo4j)
  - 3.2k FindZebra hard cases + 1.3k ReDis-QA
- **Training compute**: A100 80GB on Vertex AI, ~2 hours, ~$25
- **Native Gemma 4 features**: `<|channel>thought` thinking (toggleable), 128K context, system role

## Variants

| Repo | Format | Size | Use case |
|---|---|---|---|
| **[Raras-AI/araras-gemma4-e4b-v4-sota](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-sota)** | PEFT LoRA adapter | 147 MB | Apply to base Gemma 4 E4B |
| [Raras-AI/araras-gemma4-e4b-v4-gguf](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf) | GGUF Q4_K_M / Q5_K_M | 5.3 / 5.7 GB | llama.cpp, Ollama, edge devices |
| [Raras-AI/araras-hpo-brasil](https://huggingface.co/Raras-AI/araras-hpo-brasil) | sentence-transformers | 340 MB | PT-BR clinical → HPO matcher (companion) |
| [Raras-AI/araras-hpo-brasil-int8](https://huggingface.co/Raras-AI/araras-hpo-brasil-int8) | ONNX INT8 | 85 MB | Edge HPO matching |

## End-to-end pipeline

```
PT-BR free text (laudo, prontuário, transcription)
    ↓
[1] araras-hpo-brasil — normalize regional/lay PT-BR ("amarelão", "bebê molinho") → HPO codes
    ↓
[2] araras-gemma4-e4b — HPO-augmented prompt → ranked differential
    ↓
[3] Canonical ORPHA lookup — disease name → ORPHA code (fixes sparse-token hallucination)
    ↓
[4] PCDT overlay — ORPHA → 24 MS PCDTs → CEAF medication → reference center
    ↓
Output: Dx + evidence + SUS conduta + cost prior + nearest center
```

Total stack footprint: 5.5 GB. Runs offline on iPhone 16 Pro, Android Pixel 8, or any laptop.

## Evaluation — RareBench-BR L5_realsus

We built [RareBench-BR](https://github.com/rarasAI/rarebench-br), the first benchmark that scores diagnosis + SUS conduta simultaneously in Brazilian Portuguese. Layer L5_realsus contains 267 synthetic cases derived from **52,343 real anonymized SUS APAC trajectories** (CNS-linked, 2017-2020+), covering 12 CEAF-dispensed rare diseases.

Subsample bench (24 cases, 2 per disease), Apple M4 Pro Metal, Gemma 4 E4B Q4_K_M, `temp=1.0, top_p=0.95, top_k=64, repeat_penalty=1.15`:

| Metric | Gemma 4 raw | **With pipeline** |
|---|---:|---:|
| R@1 strict ORPHA-code match | 0% | **87.5%** |
| R@3 strict | 0% | **87.5%** |
| **Track B — PCDT-correct medication** | n/a | **100% (22/22)** |
| Latency p50 | 6.1s | 7.3s |

Track B asks: *given the top-1 dx, does the medication the pipeline would recommend match what CEAF actually dispenses in our 52k-trajectory APAC sample?* The 100% rate (22 of 22 evaluable cases) is the headline result — no other model can do this because no other model was trained on real Brazilian dispensation patterns.

The 3 dx misses (out of 24) are all SMA subtype confusion (genuine clinical ambiguity without genetic typing).

Baseline for comparison: our prior Qwen3.5-9B fine-tune ([rarasnet-diagnostic](https://huggingface.co/Raras-AI/rarasnet-diagnostic)) achieves R@1 = 16.6% on L1 at 64s p50 latency.

## How to use

### Option A: llama.cpp (recommended for inference)

```bash
hf download Raras-AI/araras-gemma4-e4b-v4-gguf araras-gemma4-e4b-v4-Q4_K_M.gguf --local-dir ./
llama-server -m araras-gemma4-e4b-v4-Q4_K_M.gguf -ngl 99 -c 8192 --jinja --reasoning off

curl http://127.0.0.1:8080/v1/chat/completions \
  -d '{
    "messages": [{"role":"user","content":"Menina 8a, hepatoesplenomegalia, opacidade corneana, atraso cognitivo, irma com mesmo quadro. Top diferenciais?"}],
    "temperature": 1.0, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.15
  }'
```

### Option B: transformers + PEFT

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = "unsloth/gemma-4-E4B-it"
adapter = "Raras-AI/araras-gemma4-e4b-v4-sota"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model.eval()

messages = [
    {"role": "system", "content": "Você é ARARAS, copiloto clínico de doenças raras em PT-BR."},
    {"role": "user", "content": "<your clinical case>"},
]
inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(inputs, max_new_tokens=400, temperature=1.0, top_p=0.95, top_k=64)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### Option C: full pipeline (BioLORD + Gemma + ORPHA lookup + PCDT)

```bash
git clone https://github.com/rarasAI/araras-gemma4
cd araras-gemma4 && pip install -r requirements.txt
python -m araras_gemma4.pipeline "Menina 8a, hepatoesplenomegalia, opacidade corneana, atraso cognitivo, irma com mesmo quadro"
```

## Limitations & responsible use

- **Never diagnoses, never prescribes.** Always informative: "your symptoms are compatible with X, here's the PCDT, here's the reference center, talk to your doctor."
- **ORPHA-code hallucination** is a known issue with all rare-disease LLMs (sparse tokens). We provide a canonical ORPHA lookup post-processor that resolves it; do NOT rely on the model's emitted ORPHA codes directly.
- **Coverage**: heavily skewed toward the 12 rare diseases that have CEAF coverage in Brazil. The long tail of 7,000 rare diseases is underrepresented.
- **Sub-types**: distinguishing SMA type 1 vs type 2 vs type 3 from phenotype text alone is genuinely hard (no model in our bench got it right consistently). Genetic typing is required.
- **Not for emergency triage.** This is a longitudinal-care copilot for chronic rare-disease patients, not an emergency-room tool.

## Citation

```bibtex
@misc{raras_gemma4_2026,
  author = {Raras Team},
  title  = {Araras-Gemma4-E4B: an open Gemma 4 fine-tune for Brazilian rare-disease care},
  year   = {2026},
  month  = {may},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-sota},
  howpublished = {Submitted to Gemma 4 Good Hackathon},
}
```

## Acknowledgments

Built on the work of Google DeepMind (Gemma 4), Unsloth (training recipe), FremyCompany (BioLORD-2023), Chen et al. (RareBench), Conitec / Ministry of Health Brazil (PCDTs), and the entire rare-disease open-science community.

Most importantly: built **for** the 13 million Brazilians the existing AI doesn't know exist, **by** one of them.
