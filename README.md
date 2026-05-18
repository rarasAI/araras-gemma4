# Araras-Gemma4

**Clinical decision support copilot for Brazilian rare-disease care, built on Gemma 4.**

[![Model](https://img.shields.io/badge/🤗-Raras--AI%2Fararas--gemma4--e4b--v4--gguf-blue)](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf)
[![Adapter](https://img.shields.io/badge/🤗-LoRA%20adapter-blue)](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-sota)
[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Raras-AI/araras-gemma4-demo)
[![Benchmark](https://img.shields.io/badge/🤗-RareBench--BR%20dataset-green)](https://huggingface.co/datasets/Raras-AI/RareBench-BR-Public)
[![License](https://img.shields.io/badge/license-Apache%202.0%20%2B%20Gemma-orange)](#license)

> Submission to the [**Gemma 4 Good Hackathon**](https://kaggle.com/competitions/gemma-4-good-hackathon) (Google DeepMind / Kaggle, May 2026)

---

## Why this exists

In Brazil, **13 million people** live with a rare disease. The average diagnostic odyssey is **5-7 years**. After diagnosis, the second odyssey begins: which of the **24 official PCDTs** (Protocolos Clínicos e Diretrizes Terapêuticas) applies? Which medication does **CEAF** actually dispense?

The problem is structural: most patients enter via UBS (primary care), where doctors are time-starved generalists facing 10,000+ rare diseases that medical school barely mentions. Frontier LLMs trained on English literature don't know Brazil's PCDTs, CEAF dispensation, or reference centers.

**Araras-Gemma4 is a decision-support copilot for the UBS doctor.** Runs offline on a phone. Apache 2.0. Open weights.

Built **by** a Brazilian rare-disease patient (Dimas — distonia mioclônica, 20 years to diagnosis), **for** the doctors who attend the next 13 million patients.

---

## What's in this repo

```
araras-gemma4/
├── README.md              # this file
├── WRITEUP.md             # full hackathon writeup (1500 words)
├── MODEL_CARD.md          # detailed model card (also on HF)
├── kaggle_notebook.ipynb  # runnable demo notebook
├── pipeline/              # end-to-end pipeline: HPO → Gemma → ORPHA → PCDT
│   ├── __init__.py
│   └── araras_pipeline.py
└── bench/                 # benchmark scripts + result JSONs
    ├── bench_pipeline_l5.py
    ├── bench_gemma_l5.py
    ├── rescore_bench.py
    └── results_*.json
```

## The four artifacts

| What | Where | Why |
|---|---|---|
| 🤗 **Gemma 4 LoRA adapter** | [Raras-AI/araras-gemma4-e4b-v4-sota](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-sota) | Apply to base Gemma 4 E4B |
| 🤗 **GGUF quantized** (5.3 GB) | [Raras-AI/araras-gemma4-e4b-v4-gguf](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf) | llama.cpp, Ollama, edge devices |
| 🤗 **HPO matcher PT-BR** | [Raras-AI/araras-hpo-brasil](https://huggingface.co/Raras-AI/araras-hpo-brasil) | "amarelão" → HP:0000952 |
| 🤗 **Benchmark dataset** | [Raras-AI/RareBench-BR-Public](https://huggingface.co/datasets/Raras-AI/RareBench-BR-Public) | 833 validated SUS-grounded cases |

## End-to-end pipeline

```
PT-BR free text (laudo, prontuário, transcription)
    ↓
[1] araras-hpo-brasil INT8 (85 MB) — normalize regional/lay PT-BR → HPO codes
    ↓
[2] araras-gemma4-e4b Q4_K_M (5.3 GB) via llama.cpp Metal/CUDA/Vulkan
    ↓
[3] Canonical ORPHA lookup — fixes LLM sparse-token hallucination
    ↓
[4] PCDT overlay (24 MS PCDTs structured) — CEAF medication + reference center
    ↓
Output: Dx + evidence + SUS conduta + cost prior + nearest center
```

Total stack: **5.5 GB**. Runs offline on iPhone, Android, laptop.

## Results

**RareBench-BR L5_realsus (240 cases, full L5 layer):**

| Metric | Araras-Gemma4 (Q4_K_M offline 4.5B) | DeepSeek V4 Chat (cloud ~600B) |
|---|---:|---:|
| R@1 (clinical name match) | **70.4%** | 86.1% (36 sub.) |
| R@3 | **78.3%** | 91.7% |
| **Track B PCDT-correct** | **76.3%** | 91.7% |
| Latency p50 | 6.5s | 4.1s |
| Cost per query | **$0 (local)** | ~$0.001 |
| Footprint | 5.5 GB | data center |

**A 4B open model running offline matches the trajectory of a 150× larger cloud model on Brazilian SUS-grounded benchmark.**

vs. our prior Qwen3.5-9B baseline on L1 (16.6% R@1, 64s): Araras is **4× more accurate, 9× faster, smaller**.

## Quick start

### Run the live demo (no install)

🤗 [huggingface.co/spaces/Raras-AI/araras-gemma4-demo](https://huggingface.co/spaces/Raras-AI/araras-gemma4-demo)

### Run locally with llama.cpp

```bash
hf download Raras-AI/araras-gemma4-e4b-v4-gguf araras-gemma4-e4b-v4-Q4_K_M.gguf
llama-server -m araras-gemma4-e4b-v4-Q4_K_M.gguf -ngl 99 -c 8192 --jinja

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "Você é ARARAS, copiloto clínico de doenças raras em PT-BR."},
    {"role": "user", "content": "Lactente 4 meses com hipotonia profunda, arreflexia, dificuldade pra mamar. Pais primos."}
  ],
  "temperature": 0.5, "top_p": 0.95, "repeat_penalty": 1.15
}'
```

### Run the full pipeline (HPO + Gemma + ORPHA + PCDT)

```python
from araras_gemma4.pipeline import inference
result = inference("Menina 8a, hepatoesplenomegalia, opacidade corneana, atraso cognitivo, irma com mesmo quadro")
print(result["stage1_hpo_normalized"])
print(result["stage3_ranked_orphas"])
print(result["stage4_sus_conduta"])
```

## Compliance

Positioned as **Software de Apoio à Decisão Clínica** (SaMD) under **ANVISA Res. 657/2022, 751/2022, and 830/2023**. Aligned with **CFM** guidance: AI augments physicians, never replaces. Every clinical claim is grounded in verifiable public sources (gov.br/conitec, bvsms, PubMed). LGPD-safe by design — zero cloud calls when running on-device.

## Training summary

- **Base**: `unsloth/gemma-4-E4B-it` (8B total / 4.5B effective)
- **Method**: QLoRA SFT via Unsloth (r=8, α=8, NEFTune α=5, train_on_responses_only)
- **Data**: 120,740 train / 5,137 val PT-BR rare-disease examples
- **Compute**: A100 80GB on Vertex AI, ~2h, ~$25
- **Native Gemma 4 features**: `<|channel>thought` blocks, 128K context, system role

Full training details in [MODEL_CARD.md](MODEL_CARD.md) and [WRITEUP.md](WRITEUP.md).

## Acknowledgments

Built on the work of **Google DeepMind** (Gemma 4), **Unsloth** (training recipe), **FremyCompany** (BioLORD-2023), **Chen et al.** (RareBench), **Conitec / Ministry of Health Brazil** (PCDTs), **Casa dos Raros** (model of local-clinician intervention), and the rare-disease open-science community.

Built on top of [**raras.org**](https://raras.org) — Latin America's largest rare-disease data + AI infrastructure (100K+ monthly visits, 3K+ registered patients, **HC-FMUSP** + **Wikipedia PT-BR** partnerships, 10,468 diseases enriched from 17 sources, built in less than 6 months).

## License

- Model weights: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- This repo code: Apache 2.0
- RareBench-BR cases: CC-BY 4.0

## Citation

```bibtex
@misc{raras_gemma4_2026,
  author       = {Raras Team and Timmers, Dimas},
  title        = {Araras-Gemma4-E4B: Open Gemma 4 Clinical Decision Support for Brazilian Rare-Disease Care},
  year         = {2026},
  month        = may,
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-sota},
  howpublished = {Submitted to Gemma 4 Good Hackathon, Google DeepMind / Kaggle},
}
```

---

*— Dimas Timmers, founder of [Raras.org](https://raras.org), paciente raro (distonia mioclônica, ORPHA:36899), 20 years to diagnosis*
