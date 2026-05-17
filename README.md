# Araras-Gemma4 — open Gemma 4 fine-tune for Brazilian rare-disease care

[![Model](https://img.shields.io/badge/🤗-Raras--AI%2Fararas--gemma4--e4b--v4-blue)](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf)
[![Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Raras-AI/araras-gemma4-demo)
[![Benchmark](https://img.shields.io/badge/RareBench--BR-1,605%20cases-green)](https://github.com/rarasAI/rarebench-br)
[![License](https://img.shields.io/badge/license-Gemma%2BApache--2.0-orange)](#license)

**The first Gemma 4 fine-tune for rare diseases in Brazilian Portuguese, grounded in real SUS dispensation data.**

> Submitted to the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon).

## Why

I'm a rare disease patient. It took me twenty years to get my diagnosis.

Brazil has 13 million rare disease patients. The average diagnostic odyssey is 5-7 years. After diagnosis, the second odyssey starts: which medication does CEAF (Componente Especializado da Assistência Farmacêutica) *actually dispense*? Which of the 24 official PCDTs applies? Which reference center is reachable?

Frontier LLMs trained on English literature don't know Brazil. **Araras does.**

— Dimas, founder, paciente raro

## What's in this repo

- [`models/`](models/) — links to HuggingFace artifacts (gguf, adapter, model card)
- [`pipeline/`](pipeline/) — end-to-end inference: BioLORD-HPO-Brasil → Gemma 4 → ORPHA lookup → SUS conduta overlay
- [`training/`](training/) — Unsloth QLoRA training recipe + dataset build scripts
- [`bench/`](bench/) — eval harness for RareBench-BR
- [`docs/`](docs/) — write-up, training log, design decisions

## Quick start

```bash
# Download model (5.3 GB)
pip install huggingface_hub
hf download Raras-AI/araras-gemma4-e4b-v4-gguf araras-gemma4-e4b-v4-Q4_K_M.gguf --local-dir ./

# Serve with llama.cpp (Metal/CUDA/Vulkan/CPU)
llama-server -m araras-gemma4-e4b-v4-Q4_K_M.gguf -ngl 99 -c 8192 --jinja --reasoning off

# Query
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Menina 8a, hepatoesplenomegalia, opacidade corneana, atraso cognitivo, irma com mesmo quadro"}],
    "temperature": 1.0, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.15
  }'
```

### Full pipeline (BioLORD + Gemma + ORPHA + PCDT)

```python
from araras_pipeline import inference
result = inference("Menina 8a, hepatoesplenomegalia, opacidade corneana bilateral, atraso cognitivo, facies grosseira, irma com mesmo quadro")
print(result["stage1_hpo_normalized"])  # PT-BR → HPO codes via araras-hpo-brasil
print(result["stage3_ranked_orphas"])   # ORPHA codes resolved from disease names
print(result["stage4_sus_conduta"])     # PCDT + CEAF medication for the top-1 dx
```

## Models

| Repo | What | Size | Purpose |
|---|---|---|---|
| [Raras-AI/araras-gemma4-e4b-v4-gguf](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf) | Quantized GGUF | 5.3 GB (Q4_K_M) | Edge / mobile deployment |
| [Raras-AI/araras-gemma4-e4b-v4-sota](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-sota) | LoRA adapter | 147 MB | Apply to base Gemma 4 E4B |
| [Raras-AI/araras-hpo-brasil](https://huggingface.co/Raras-AI/araras-hpo-brasil) | BioLORD fine-tune | 340 MB | PT-BR clinical → HPO |
| [Raras-AI/araras-hpo-brasil-int8](https://huggingface.co/Raras-AI/araras-hpo-brasil-int8) | ONNX INT8 | 85 MB | Edge HPO matching |

## Results

**RareBench-BR L5_realsus** — benchmark anchored in **52,343 real APAC trajectories** from DataSUS, 12 CEAF-dispensed rare diseases.

**End-to-end pipeline (BioLORD-HPO-Brasil + Gemma 4 + ORPHA lookup + PCDT overlay):**

| Metric | Pipeline | Gemma 4 raw | Qwen3.5-9B baseline |
|---|---:|---:|---:|
| R@1 strict ORPHA-code | **87.5%** | 0% (hallucinated codes) | n/a |
| R@3 strict | **87.5%** | 0% | n/a |
| **🔥 Track B — PCDT-correct medication** | **100% (22/22)** | n/a | n/a |
| Latency p50 | **7.3s** | 6.1s | 64s |
| RAM footprint | 5.5 GB | 5.3 GB | 6.5 GB |
| Inference cost | $0 (local) | $0 | $0 |

The **100% Track B** rate is the headline: every evaluable diagnosis was followed by a medication recommendation matching what CEAF actually dispenses for that ORPHA. No other model does this — no other model was trained on real Brazilian dispensation patterns from 52k linked APAC trajectories.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  INPUT: free-text PT-BR (laudo, prontuário, transcription)     │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│  [1] araras-hpo-brasil INT8 (85 MB)                             │
│       PT-BR clinical idiom → HPO codes                          │
│       "amarelão" → HP:0000952 (Jaundice)                        │
│       "bebê molinho" → HP:0001252 (Hypotonia)                   │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│  [2] araras-gemma4-e4b Q4_K_M (5.3 GB) via llama.cpp            │
│       HPO-augmented prompt → ranked differential                │
│       Native <|channel>thought blocks (configurable)            │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│  [3] Canonical ORPHA lookup (post-processor)                    │
│       disease name → ORPHA code (fixes LLM hallucination)       │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│  [4] PCDT overlay (24 MS protocols, structured YAML)            │
│       ORPHA → PCDT → CEAF medication → reference center         │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│  OUTPUT: Dx + evidence + SUS conduta + cost prior + center      │
└────────────────────────────────────────────────────────────────┘
```

Total footprint **~5.5 GB**. Zero cloud. Runs on iPhone 16 Pro / Android Pixel 8 / any laptop.

## Training

- **Base**: `unsloth/gemma-4-E4B-it` (8B total / 4.5B effective)
- **Method**: QLoRA SFT, r=8, α=8, dropout=0, NEFTune α=5, `train_on_responses_only`
- **Data**: 120,740 train / 5,137 val PT-BR rare-disease examples (deduplicated)
- **Compute**: ~2h on A100 80GB via Vertex AI, ~$25 total
- **Recipe**: see [`training/train_gemma4_pipeline.py`](training/train_gemma4_pipeline.py)

## Citation

```bibtex
@misc{raras_gemma4_2026,
  author = {Raras Team},
  title  = {Araras-Gemma4-E4B: an open Gemma 4 fine-tune for Brazilian rare-disease care},
  year   = {2026},
  month  = {may},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf},
}
```

## License

- Model weights: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- This repo (code, scripts, benchmark): Apache 2.0
- RareBench-BR cases: CC-BY 4.0

## Acknowledgments

Built on the work of Google DeepMind (Gemma 4), Unsloth (training recipe), FremyCompany (BioLORD-2023), Chen et al. (RareBench), Conitec / Ministry of Health Brazil (PCDTs), and the entire rare-disease open-science community.

Most importantly: built for the 13 million Brazilians the existing AI doesn't know exist.
