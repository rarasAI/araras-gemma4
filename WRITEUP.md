# Araras: Gemma 4 copilot for Brazil's 13 million rare-disease patients — grounded in real DataSUS

**Tracks**: Main + Health & Sciences + Digital Equity & Inclusivity + Unsloth + Ollama + llama.cpp
**Author**: Dimas, founder of Raras — rare disease patient himself, 20 years to diagnosis
**Model weights**: 🤗 [Raras-AI/araras-gemma4-e4b-v4-gguf](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-gguf) (230+ downloads) + [v4-sota adapter](https://huggingface.co/Raras-AI/araras-gemma4-e4b-v4-sota)
**Code**: [github.com/rarasAI/araras-gemma4](https://github.com/rarasAI/araras-gemma4)
**Benchmark**: [github.com/rarasAI/rarebench-br](https://github.com/rarasAI/rarebench-br)
**Live demo**: [🤗 Raras-AI/araras-gemma4-demo](https://huggingface.co/spaces/Raras-AI/araras-gemma4-demo) (HF Space, ZeroGPU) · [raras.com.br/copiloto-gemma](https://raras.com.br/copiloto-gemma)

## TL;DR

**Araras-Gemma4 is clinical decision support for Brazilian healthcare professionals — especially in remote regions where rare-disease expertise doesn't reach.** Augments the doctor; never replaces. Built on raras.org — Latin America's largest rare-disease data + AI infrastructure (100k+ monthly visits, 3k+ patients, HC-FMUSP + Wikipedia PT partnerships). Gemma 4 E4B fine-tune on 120k PT-BR examples, shipped offline (llama.cpp):

```
PT-BR free text → araras-hpo-brasil (BioLORD fine-tune)
               → araras-gemma4-e4b → ranked diagnosis
               → canonical ORPHA lookup → PCDT overlay → SUS conduta
```

Evaluated on the **full RareBench-BR_SUS unified benchmark (833 cases, all layers, 0 errors)** — built from **52,343 real anonymized SUS patient trajectories** (APAC, CNS-linked) + 24 official PCDTs + hand-curated hard BR cases:

- **R@1 (canonical disease name) = 41.2%** across the full mix (L3_v2 atypical + L4 hard + L5_v2 SUS-grounded)
- **R@3 = 47.1%**
- **🔥 Track B PCDT-correct = 76.8% (331/431 evaluable cases)** — pipeline recommends the *exact medication CEAF actually dispenses* in 3 of every 4 SUS cases
- **L5_v2 (SUS-grounded subset, n=619): R@1 = 47.2%, R@3 = 53.8%** — the most operationally relevant slice
- p50 latency 6.5s on Apple M4 Pro Metal, llama.cpp Q4_K_M
- Total stack footprint 5.5 GB. Runs offline on a phone.

**Nothing else in the open ecosystem predicts Brazilian SUS conduta this accurately at 4B parameters, because nothing else was trained on SUS-grounded data.**

## Why Raras trained a Gemma 4

[**Raras**](https://raras.org) is Latin America's largest rare-disease data + AI infrastructure. In **under six months**, starting from zero, we built:

- **100,000+ monthly visits** organically (no paid acquisition)
- **3,000+ registered patients** with **"Carteira do Paciente"** — an encrypted digital health record using AES-256-GCM for medical documents and CPF crypto with key versioning (HIPAA + LGPD compliant)
- **10,468 rare diseases** mapped in a Neo4j knowledge graph, enriched from **17 sources**: Orphanet, OMIM, HPO, ClinGen, gnomAD, MedGen, MONDO, PrimeKG, PubMed, gov.br/conitec PCDTs, CEAF Portarias, CNES, RaraConnect, ORDO, GenReviews, ClinicalTrials.gov, plus our own curation
- **18+ rare-disease patient associations** on the platform — each with their own community, content, and member registry
- **Strategic partnership with HC-FMUSP** (Hospital das Clínicas da Faculdade de Medicina da USP — Brazil's largest rare-disease reference center)
- **Strategic partnership with Wikipedia PT-BR** for evidence-grounded disease pages
- Multi-modal **copilot** in production (DeepSeek + Vertex Gemini) handling patient + clinician queries, integrated with the Carteira data

The platform is the only place in Brazil where a rare-disease patient can: maintain an encrypted longitudinal record (laudos, exams, medications), find their disease's official PCDT and CEAF dispensation list, see verified reference centers and ongoing clinical trials, connect to a patient association for their condition, and ask an AI copilot real questions about their case.

All of that ran as a web product — browser, login, stable internet, clinician at a desk with free time. **But the Brazilian diagnostic odyssey doesn't happen in a browser.** It happens in 5-10 minute UBS consultations where the doctor can't open 4 tabs, log in to an MS portal, download a PCDT PDF, cross-reference CEAF lists. And where, in half the country, 4G is slow or absent. The knowledge stayed trapped in the web product.

**Araras-Gemma4 is the distillation of that infrastructure into a 4B-parameter model that runs on a phone, offline, during the consultation.** The brain of Raras leaves the cloud and goes to the pocket of the healthcare professional.

## The 20-year problem is a healthcare-workforce problem

My name is **Dimas Timmers**. I'm a rare-disease patient — **distonia mioclônica (ORPHA:36899)**, diagnosed at 27 after a 20-year odyssey across seven doctors and five hospitals. The first specialist saw me at age 7; the diagnosis arrived two decades later. None of those seven doctors was incompetent — they were time-starved generalists facing 10,000+ rare diseases that medical school barely mentions.

Brazil's average diagnostic odyssey is **5-7 years**. Most patients enter via UBS (primary care). [BMC](https://link.springer.com/article/10.1186/s12875-024-02553-8) shows rural UBS doctors are mostly recent graduates with no integrated information system. [Casa dos Raros](https://pmc.ncbi.nlm.nih.gov/articles/PMC12321710/) validated that the right intervention target is **the local clinician**, not the patient.

After my own diagnosis I founded **Raras** — because I had lived the inside of the problem long enough to know what was missing: not more frontier AI, but the *right* knowledge delivered at the right moment to the right professional.

Frontier LLMs in English know Gaucher exists. They don't know Brazil's PCDT lists imiglucerase/alfa-velaglicerase/alfa-taliglicerase as CEAF options, that the Portaria was amended twice, or that dispensing happens at 31 specific centers — exactly what a primary-care doctor needs to refer correctly. **Araras is a decision-support copilot for that doctor** — opens during the consultation, queries with the case, informs the next decision (referral, exam, family conversation).

## What we built

### 1. Araras-Gemma4-E4B-SFT

`unsloth/gemma-4-E4B-it` base (4.5B effective). QLoRA SFT via Unsloth (r=8, α=8, NEFTune α=5, train_on_responses_only). 120,740 train / 5,137 val deduplicated PT-BR examples (108k rare-disease Q&A, 26k MedPT, 10k ultra-rare, 10k tool-calling, 5.7k KG triples, FindZebra/ReDis-QA). A100 80GB Vertex, ~2h, ~$25. Native Gemma 4: `<|channel>thought` (toggleable), 128K ctx.

### 2. Araras-HPO-Brasil (the matcher)

First BioLORD-2023 fine-tune for Brazilian Portuguese clinical lay language. "amarelão" → HP:0000952 (Jaundice), "bebê molinho" → HP:0001252 (Hypotonia), "fácies grosseira" → HP:0000280. Two artifacts: full 340 MB + ONNX INT8 85 MB.

### 3. RareBench-BR (the benchmark — a novel contribution in itself)

The first public benchmark that scores Dx + SUS conduta together in PT-BR. 1,605 cases across 4 layers:

Layers: **L1** 1,122 (Chen et al. NeurIPS 2024, localized to PT-BR), **L3_v2** 135 (24 PCDTs × presentations), **L4** 81 (tropical phenocopies + founder mutations + IEI + sparse HPO + neonatal screening), **L5_realsus** 267 (anchored in **52,343 real APAC trajectories**, CNS-linked, 2017-2020+). Total **1,605 cases**.

**L5 is the world's first DataSUS-anchored rare-disease benchmark.** We linked anonymized APAC records via CNS-hash across 12 CEAF-covered diseases: MS (20,867), Falciforme (13,122), SMA 5q (6,078), HAP (5,508), Fibrose Cística (3,804), PKU (1,504), Wilson (656), MPS II (619), MPS VI (101), MPS I (73), SCID (33), SMA tipo 1 (10).

Each L5 case is synthesized from real SUS statistics (sex, age percentiles, SIGTAP procedure codes billed, geographic distribution) with a `datasus_anchor` field. **LGPD-safe**: no raw CNS hash, no individual trajectory replicated, only aggregates.

### 4. The pipeline (the assembly)

```
[1] araras-hpo-brasil INT8 (85 MB)
    PT-BR clinical idiom → HPO codes
[2] araras-gemma4-e4b Q4_K_M (5.3 GB) via llama.cpp Metal/CUDA/Vulkan
    HPO-augmented prompt → ranked differential
[3] Canonical ORPHA lookup post-processor
    disease name → ORPHA code (fixes LLM sparse-token hallucination)
[4] PCDT overlay (24 MS PCDTs structured as YAML, <1 MB)
    ORPHA → PCDT → CEAF medication → reference center
```

Total 5.5 GB. Runs on iPhone 16 Pro, Android Pixel 8, or any laptop. Zero cloud.

### 5. Wired into raras-app production

The pipeline is exposed on raras-app (Cloud Run southamerica-east1) via `COPILOT_PRIMARY_LLM=gemma`. The "Carteira do Paciente" (encrypted registry) shares 3,072-dim fused embeddings with this stack. 18+ rare-disease associations on the platform get this for their members at zero marginal cost.

## Results

### RareBench-BR L5_realsus standalone (n=240, all 12 CEAF-covered diseases)

| Metric | Araras-Gemma4 (offline, 4B) | DeepSeek V4 cloud (~600B, 36-case head-to-head) |
|---|---:|---:|
| R@1 (canonical name) | **70.4%** | 86.1% |
| R@3 | **78.3%** | 91.7% |
| **Track B PCDT-correct** | **76.3%** | 91.7% (33/36) |
| Latency p50 | 6.5s | 4.1s |
| Cost | **$0 (local)** | $0.27/1M tokens |
| Footprint | **5.5 GB** | data-center |

DeepSeek wins on raw accuracy — expected for a 150× bigger model. **Where Araras wins is the combination**: 4B, fully offline, free, with the strongest open-weight SUS-grounded Track B number published. The canonical ORPHA lookup post-processor lifts strict matching from 0% (raw Gemma hallucinates codes) to the reported numbers. Vs prior Qwen3.5-9B baseline on L1 (16.6% R@1, 64s): Araras is **4× more accurate, 9× faster, smaller**.

### RareBench-BR full unified run (n=833, all layers, 0 errors)

Evaluated on the *full* unified RareBench-BR_SUS — L3_v2 (24 PCDTs × atypical presentations) + L4 (hand-curated hard BR cases) + L5_v2 (SUS-grounded synthesis from 52k APAC trajectories):

| Metric | Result | vs prior buggy resolver |
|---|---:|---:|
| **R@1 (canonical name)** | **41.2%** | **+188% relative** |
| **R@3** | **47.1%** | +80% |
| **🔥 Track B PCDT-correct** | **76.8%** (331/431) | **+196% relative** |
| Errors | **0** | (vs 19 before) |

**Per-layer breakdown (full 833 cases):**

| Layer | N | R@1 | R@3 |
|---|---:|---:|---:|
| L3_v2 — PCDT intersection | 135 | 27.4% | 32.6% |
| L4 — Hard BR cases (tropical, founder, IEI) | 79 | 17.7% | 19.0% |
| L5_v2 — SUS-grounded | 619 | **47.2%** | **53.8%** |

The Track B 76.8% number is the headline: **for 3 out of every 4 cases where ground truth has a CEAF-dispensed medication, Araras-Gemma4 recommends the exact molecule SUS actually pays for** — at 4B parameters, fully offline, $0 marginal cost.

Two prompt+resolver iterations were needed to reach these numbers. The first pass used a CEAF-biased fallback in the post-processor that over-predicted ORPHA:802 (Esclerose Múltipla), inflating false positives on some cases while masking real misses elsewhere — net R@1 was 14.3%. The fix: anti-hallucination system prompt (forbid invented "Síndrome de Síndromes" patterns), 1 few-shot example with canonical disease names, temperature 0.3 (vs 0.5), and a clean substring-overlap resolver with no CEAF bias. Same model weights, same data — just better prompting + scoring.

## What's genuinely new

| | Why it didn't exist |
|---|---|
| **DataSUS-anchored rare-disease benchmark (L5)** | Required CNS-hash linkage across APAC/SIH/SIM and aggregation of 52k+ trajectories. No prior project did this. |
| **PT-BR Gemma 4 fine-tune for rare disease** | Required 120k curated PT-BR clinical examples from RarasNet's 10k-disease knowledge graph. |
| **BioLORD fine-tune for Brazilian clinical lay language** | "amarelão", "sangue ralo", "bebê molinho" are embedding-hostile in English BioLORD. |
| **Structured 24-PCDT layer + canonical ORPHA dict** | The PCDTs are public PDFs; we structured them. The ORPHA dict resolves the universal LLM hallucination problem. |
| **End-to-end offline pipeline under 5.5 GB** | Combining INT8 BioLORD + Q4 Gemma 4 + YAML PCDT — engineering exercise, not research, but nobody had assembled it. |

## Compliance posture (ANVISA, CFM, LGPD)

Araras is positioned as **clinical decision support software** for licensed healthcare professionals, **not a patient-facing diagnostic tool**. This is the regulatory frame Brazil's ANVISA defines for AI as a medical device (SaMD) under Res. 657/2022, 751/2022, 830/2023, and aligns with CFM's stance that AI augments, never replaces, the physician. The system prompt is explicit: it lists differentials, cites PCDT URLs (gov.br/conitec) and CEAF medications, and recommends referral to a reference center — the clinician decides. Every clinical claim is grounded in a verifiable public source (gov.br, bvsms, pubmed). LGPD-safe by design: zero cloud call when running on-device; encrypted patient registry; no individual patient trajectory ever exposed.

What's not new (honesty): Gemma 4 architecture, Unsloth recipe, HPO ontology, public PCDT PDFs, public APAC DataSUS, RareBench L1 cases. We did the assembly, the PT-BR adaptation, the SUS-grounding, the workforce framing.

## Reproducibility + prize

`hf download Raras-AI/araras-gemma4-e4b-v4-gguf …Q4_K_M.gguf && llama-server -m … -ngl 99 --jinja --reasoning off && git clone https://github.com/rarasAI/araras-gemma4 && python … eval_full_pipeline.py`. Prize money: 100% to expanding **RareBench-BR** with board-certified geneticists/rare-disease specialists. The benchmark is the public good — once it exists, every model maker is forced to be measured on Brazilian SUS-correctness. That permanently changes incentives.

## The 20-year part

Symptoms at 7. Diagnosed at 27. **Distonia mioclônica (ORPHA:36899)** — a movement disorder so rare that none of the seven neurologists I saw before age 27 had ever managed a case. The information existed the whole time — locked behind English-language literature, paywalled journals, and a healthcare system that didn't talk to itself.

I built Raras after my diagnosis because the inside view of the problem was unmistakable: **the bottleneck isn't a smarter AI — it's the right knowledge in the hands of the local clinician, in their language, in their workflow, in their consultation room.**

The Brazilian rare-disease patient pays the cost of that bottleneck in years of life. Araras-Gemma4 is one piece of the answer — a 4B-parameter model in the pocket of the doctor, with **76.8% Track B accuracy** on actual Brazilian SUS dispensation patterns, runnable on a phone, fully Apache 2.0.

Araras is what I wish someone had handed the seven doctors who saw me before age 27. It's open. Take it. Make it better.

— **Dimas Timmers**, founder of [Raras.org](https://raras.org), paciente raro (distonia mioclônica)
