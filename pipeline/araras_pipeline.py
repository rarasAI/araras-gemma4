"""Araras pipeline: BioLORD-HPO-Brasil → Gemma 4 → ORPHA lookup → PCDT overlay.

End-to-end inference:
  free-text PT-BR
    → [1] araras-hpo-brasil normaliza fenótipos PT-BR → HPO codes
    → [2] araras-gemma4-e4b gera diferencial diagnóstico (com HPO no prompt)
    → [3] dicionário canônico nome→ORPHA fixa hallucination de codes
    → [4] PCDT overlay adiciona conduta SUS + fármaco CEAF real
"""
from __future__ import annotations
import json, re, time, urllib.request, unicodedata, numpy as np
from pathlib import Path

EMB_PATH = "/tmp/araras_hpo_embeddings.npz"
META_PATH = "/tmp/araras_hpo_metadata.json"
LLAMA_URL = "http://127.0.0.1:8089/v1/chat/completions"

# Canonical dict: disease name keywords → ORPHA + PCDT info (used as post-processor)
CANONICAL = {
    "esclerose multipla": {"orpha": "ORPHA:802", "pcdt_slug": "esclerose-multipla",
                            "ceaf_drugs": ["betainterferona-1a/1b", "acetato de glatirâmer", "natalizumabe", "fingolimode", "ocrelizumabe"]},
    "neuromielite optica": {"orpha": "ORPHA:71211", "pcdt_slug": None,
                            "ceaf_drugs": ["rituximabe (off-label)"]},
    "doenca falciforme": {"orpha": "ORPHA:232", "pcdt_slug": "doenca-falciforme",
                          "ceaf_drugs": ["hidroxiureia", "deferasirox", "L-glutamina"]},
    "anemia falciforme": {"orpha": "ORPHA:232", "pcdt_slug": "doenca-falciforme",
                          "ceaf_drugs": ["hidroxiureia", "deferasirox", "L-glutamina"]},
    "drepanocitose": {"orpha": "ORPHA:232", "pcdt_slug": "doenca-falciforme",
                      "ceaf_drugs": ["hidroxiureia"]},
    "atrofia muscular espinhal": {"orpha": "ORPHA:83330", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                                   "ceaf_drugs": ["nusinersena", "risdiplam", "onasemnogeno-abeparvoveque"]},
    "amiotrofia espinhal": {"orpha": "ORPHA:83330", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                             "ceaf_drugs": ["nusinersena", "risdiplam"]},
    "sma tipo 2": {"orpha": "ORPHA:83330", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                    "ceaf_drugs": ["nusinersena", "risdiplam"]},
    "sma 5q": {"orpha": "ORPHA:83330", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                "ceaf_drugs": ["nusinersena", "risdiplam"]},
    "werdnig-hoffmann": {"orpha": "ORPHA:70", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                          "ceaf_drugs": ["nusinersena", "onasemnogeno-abeparvoveque"]},
    "ame tipo 1": {"orpha": "ORPHA:70", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                    "ceaf_drugs": ["nusinersena", "onasemnogeno-abeparvoveque"]},
    "ame tipo i": {"orpha": "ORPHA:70", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                    "ceaf_drugs": ["nusinersena", "onasemnogeno-abeparvoveque"]},
    "sma tipo 1": {"orpha": "ORPHA:70", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                    "ceaf_drugs": ["nusinersena", "onasemnogeno-abeparvoveque"]},
    "spinal muscular atrophy type 1": {"orpha": "ORPHA:70", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                                        "ceaf_drugs": ["nusinersena", "onasemnogeno-abeparvoveque"]},
    "spinal muscular atrophy": {"orpha": "ORPHA:83330", "pcdt_slug": "atrofia-muscular-espinhal-5q",
                                 "ceaf_drugs": ["nusinersena", "risdiplam"]},
    "hipertensao arterial pulmonar": {"orpha": "ORPHA:182090", "pcdt_slug": "hipertensao-pulmonar",
                                       "ceaf_drugs": ["sildenafila", "bosentana", "ambrisentana", "macitentana"]},
    "hipertensao pulmonar": {"orpha": "ORPHA:182090", "pcdt_slug": "hipertensao-pulmonar",
                              "ceaf_drugs": ["sildenafila", "bosentana", "ambrisentana"]},
    "fibrose cistica": {"orpha": "ORPHA:586", "pcdt_slug": "fibrose-cistica",
                        "ceaf_drugs": ["dornase-alfa", "tobramicina inalatória", "ivacaftor", "lumacaftor-ivacaftor"]},
    "mucoviscidose": {"orpha": "ORPHA:586", "pcdt_slug": "fibrose-cistica",
                      "ceaf_drugs": ["dornase-alfa", "tobramicina inalatória"]},
    "fenilcetonuria": {"orpha": "ORPHA:716", "pcdt_slug": "fenilcetonuria",
                       "ceaf_drugs": ["fórmula metabólica isenta de fenilalanina", "sapropterina (casos selecionados)"]},
    "pku": {"orpha": "ORPHA:716", "pcdt_slug": "fenilcetonuria",
             "ceaf_drugs": ["fórmula metabólica isenta de fenilalanina"]},
    "fenilalaninemia": {"orpha": "ORPHA:716", "pcdt_slug": "fenilcetonuria",
                         "ceaf_drugs": ["fórmula metabólica isenta de fenilalanina"]},
    "hiperfenilalanin": {"orpha": "ORPHA:716", "pcdt_slug": "fenilcetonuria",
                          "ceaf_drugs": ["fórmula metabólica isenta de fenilalanina"]},
    "phenylketonuria": {"orpha": "ORPHA:716", "pcdt_slug": "fenilcetonuria",
                         "ceaf_drugs": ["fórmula metabólica isenta de fenilalanina"]},
    "doenca de wilson": {"orpha": "ORPHA:905", "pcdt_slug": "doenca-de-wilson",
                         "ceaf_drugs": ["d-penicilamina", "trientina", "zinco-acetato"]},
    "wilson disease": {"orpha": "ORPHA:905", "pcdt_slug": "doenca-de-wilson",
                        "ceaf_drugs": ["d-penicilamina", "trientina"]},
    "deficiencia de ceruloplasmina": {"orpha": "ORPHA:905", "pcdt_slug": "doenca-de-wilson",
                                       "ceaf_drugs": ["d-penicilamina", "trientina"]},
    "kayser-fleischer": {"orpha": "ORPHA:905", "pcdt_slug": "doenca-de-wilson",
                          "ceaf_drugs": ["d-penicilamina", "trientina"]},
    "mucopolissacaridose tipo i": {"orpha": "ORPHA:579", "pcdt_slug": "mucopolissacaridose-tipo-i",
                                    "ceaf_drugs": ["laronidase"]},
    "hurler": {"orpha": "ORPHA:579", "pcdt_slug": "mucopolissacaridose-tipo-i",
                "ceaf_drugs": ["laronidase"]},
    "mps i": {"orpha": "ORPHA:579", "pcdt_slug": "mucopolissacaridose-tipo-i",
               "ceaf_drugs": ["laronidase"]},
    "mucopolissacaridose tipo ii": {"orpha": "ORPHA:580", "pcdt_slug": "mucopolissacaridose-tipo-ii",
                                     "ceaf_drugs": ["idursulfase-alfa"]},
    "hunter": {"orpha": "ORPHA:580", "pcdt_slug": "mucopolissacaridose-tipo-ii",
                "ceaf_drugs": ["idursulfase-alfa"]},
    "mps ii": {"orpha": "ORPHA:580", "pcdt_slug": "mucopolissacaridose-tipo-ii",
                "ceaf_drugs": ["idursulfase-alfa"]},
    "iduronato sulfatase": {"orpha": "ORPHA:580", "pcdt_slug": "mucopolissacaridose-tipo-ii",
                             "ceaf_drugs": ["idursulfase-alfa"]},
    "iduronato-2-sulfatase": {"orpha": "ORPHA:580", "pcdt_slug": "mucopolissacaridose-tipo-ii",
                               "ceaf_drugs": ["idursulfase-alfa"]},
    "mucopolissacaridose tipo vi": {"orpha": "ORPHA:583", "pcdt_slug": "mucopolissacaridose-tipo-vi",
                                     "ceaf_drugs": ["galsulfase"]},
    "maroteaux-lamy": {"orpha": "ORPHA:583", "pcdt_slug": "mucopolissacaridose-tipo-vi",
                        "ceaf_drugs": ["galsulfase"]},
    "mps vi": {"orpha": "ORPHA:583", "pcdt_slug": "mucopolissacaridose-tipo-vi",
                "ceaf_drugs": ["galsulfase"]},
    "doenca de gaucher": {"orpha": "ORPHA:355", "pcdt_slug": "doenca-de-gaucher",
                          "ceaf_drugs": ["imiglucerase", "alfa-velaglicerase", "alfa-taliglicerase", "miglustate", "eliglustate"]},
    "doenca de pompe": {"orpha": "ORPHA:365", "pcdt_slug": "doenca-de-pompe",
                        "ceaf_drugs": ["alfa-alglicosidase"]},
    "doenca de fabry": {"orpha": "ORPHA:324", "pcdt_slug": "doenca-de-fabry",
                        "ceaf_drugs": ["alfa-agalsidase", "beta-agalsidase", "migalastate"]},
    "scid": {"orpha": "ORPHA:183660", "pcdt_slug": None,
              "ceaf_drugs": ["TCTH", "IVIg"]},
    "imunodeficiencia combinada grave": {"orpha": "ORPHA:183660", "pcdt_slug": None,
                                          "ceaf_drugs": ["TCTH"]},
    "hemoglobinuria paroxistica noturna": {"orpha": "ORPHA:447", "pcdt_slug": "hemoglobinuria-paroxistica-noturna",
                                            "ceaf_drugs": ["eculizumabe", "ravulizumabe"]},
}

def normalize_str(s):
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ─── Stage 1: BioLORD HPO normalization ──────────────────────────────────
_model = None
_embs = None
_meta = None

def load_hpo_index():
    global _model, _embs, _meta
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("Raras-AI/araras-hpo-brasil")
        _embs = np.load(EMB_PATH)["embeddings"]
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _embs, _meta

def split_phenotypes(text):
    """Naive phenotype splitter: by comma, period, newline. Drops short fragments."""
    # Remove numbers ("8 anos"), keep clinical phrases
    parts = re.split(r"[.,;\n]| e ", text)
    out = []
    for p in parts:
        p = p.strip(" -*")
        if 3 <= len(p) <= 80 and not p[0].isdigit():
            out.append(p)
    return out[:30]

def extract_hpo(text, threshold=0.65, top_k=2):
    """Extract HPO codes from PT-BR free-text using araras-hpo-brasil."""
    model, embs, meta = load_hpo_index()
    parts = split_phenotypes(text)
    if not parts:
        return []
    q = model.encode(parts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    out = []
    seen = set()
    for phrase, qv in zip(parts, q):
        sims = embs @ qv
        top = np.argsort(-sims)[:top_k]
        for idx in top:
            sim = float(sims[idx])
            if sim >= threshold:
                hpo_id = meta[idx]["hpo_id"]
                if hpo_id not in seen:
                    seen.add(hpo_id)
                    out.append({
                        "phrase_pt": phrase,
                        "hpo_id": hpo_id,
                        "hpo_name": meta[idx]["name"],
                        "similarity": round(sim, 3),
                    })
                break  # take best match only
    return out


# ─── Stage 2: Gemma 4 inference (augmented prompt) ────────────────────────

SYS = """Você é ARARAS, copiloto clínico de doenças raras em PT-BR.
Para o caso abaixo, liste o TOP-5 diferencial em ordem de probabilidade.
FORMATO (uma linha por diagnóstico):
1. Nome da doença — Justificativa breve
2. Nome da doença — Justificativa breve
...
NUNCA invente código ORPHA — escreva só o nome canônico da doença em PT-BR."""

def call_gemma(case_text, hpo_normalized):
    """Call llama-server with HPO-augmented prompt."""
    hpo_str = "\n".join(f"- {h['phrase_pt']!r} → {h['hpo_id']} ({h['hpo_name']})" for h in hpo_normalized)
    user = case_text
    if hpo_str:
        user = f"{case_text}\n\n[Fenótipos HPO já normalizados pelo araras-hpo-brasil:]\n{hpo_str}"
    body = {
        "model": "araras",
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
        ],
        "temperature": 1.0, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.15,
        "max_tokens": 700,
    }
    req = urllib.request.Request(LLAMA_URL, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=240) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"].get("content", ""), time.time() - t0


# ─── Stage 3: ORPHA lookup (post-processor) ──────────────────────────────

def resolve_orpha(model_output):
    """Match model's disease names against canonical dict → return ranked ORPHAs."""
    txt = normalize_str(model_output)
    found = []
    seen = set()
    # parse numbered list to preserve ranking
    lines = re.split(r"\n+", model_output)
    for line in lines:
        line_n = normalize_str(line)
        for keyword, info in CANONICAL.items():
            if keyword in line_n and info["orpha"] not in seen:
                found.append({
                    "orpha": info["orpha"],
                    "matched_keyword": keyword,
                    "pcdt_slug": info["pcdt_slug"],
                    "ceaf_drugs": info["ceaf_drugs"],
                })
                seen.add(info["orpha"])
                break  # one per line
    # fallback: full-text scan
    if not found:
        for keyword, info in CANONICAL.items():
            if keyword in txt and info["orpha"] not in seen:
                found.append({"orpha": info["orpha"], "matched_keyword": keyword,
                              "pcdt_slug": info["pcdt_slug"], "ceaf_drugs": info["ceaf_drugs"]})
                seen.add(info["orpha"])
    return found


# ─── Stage 4: PCDT overlay (SUS-conduta) ─────────────────────────────────

def sus_conduta(top1_orpha_info):
    """Generate SUS-conduta string from top-1 ORPHA → PCDT + CEAF."""
    if not top1_orpha_info:
        return None
    info = top1_orpha_info
    if info["pcdt_slug"]:
        return (
            f"Encaminhar paciente para centro de referência. "
            f"Solicitar exames confirmatórios. Caso confirmado, considerar terapia via CEAF "
            f"conforme PCDT '{info['pcdt_slug']}' do Ministério da Saúde. "
            f"Fármacos típicos do PCDT/CEAF: {', '.join(info['ceaf_drugs'][:3])}."
        )
    return f"Sem PCDT específico do MS. Avaliar terapias: {', '.join(info['ceaf_drugs'][:3])}."


# ─── End-to-end ─────────────────────────────────────────────────────────

def inference(case_text):
    t_total = time.time()
    t = time.time(); hpo = extract_hpo(case_text); t_hpo = time.time() - t
    t = time.time(); gemma_out, t_gemma = call_gemma(case_text, hpo); t_call = t_gemma
    t = time.time(); orphas = resolve_orpha(gemma_out); t_orpha = time.time() - t
    conduta = sus_conduta(orphas[0]) if orphas else None
    return {
        "input": case_text,
        "stage1_hpo_normalized": hpo,
        "stage2_gemma_raw": gemma_out,
        "stage3_ranked_orphas": orphas,
        "stage4_sus_conduta": conduta,
        "timings_s": {"hpo": round(t_hpo, 2), "gemma": round(t_call, 2),
                       "orpha_lookup": round(t_orpha, 3), "total": round(time.time()-t_total, 2)},
    }


if __name__ == "__main__":
    test = ("Menina de 8 anos, hepatoesplenomegalia progressiva, "
            "opacidade corneana bilateral, atraso cognitivo importante, "
            "fácies grosseira, irmã mais velha com mesmo quadro.")
    print("INPUT:", test)
    print()
    r = inference(test)
    print("[1] HPO normalized (araras-hpo-brasil):")
    for h in r["stage1_hpo_normalized"]:
        print(f"   {h['phrase_pt']!r:40} → {h['hpo_id']} {h['hpo_name']} ({h['similarity']})")
    print()
    print("[2] Gemma 4 raw output:")
    print(r["stage2_gemma_raw"][:600])
    print()
    print("[3] ORPHAs ranked (canonical lookup):")
    for o in r["stage3_ranked_orphas"][:5]:
        print(f"   {o['orpha']:>14}  (matched {o['matched_keyword']!r}, PCDT={o['pcdt_slug']})")
    print()
    print("[4] SUS Conduta:")
    print(r["stage4_sus_conduta"])
    print()
    print("Timings:", r["timings_s"])
