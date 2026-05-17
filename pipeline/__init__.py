"""Araras pipeline: BioLORD-HPO-Brasil → Gemma 4 → ORPHA lookup → SUS conduta.

Public API:
  from araras_gemma4.pipeline import inference, extract_hpo, call_gemma, resolve_orpha, sus_conduta
"""
from .araras_pipeline import (
    inference,
    extract_hpo,
    call_gemma,
    resolve_orpha,
    sus_conduta,
    CANONICAL,
)
