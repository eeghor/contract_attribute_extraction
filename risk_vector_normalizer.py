"""
risk_vector_normalizer.py
=========================
Converts a raw ``classify_contract()`` result dict into three things:

  1. **numeric_vector()** — a fixed-length float array suitable for cosine
     similarity, clustering, or archetype assignment.  Mixes LLM-sourced
     scores (from ``risk_vector_*`` keys) with deterministic signals extracted
     by ``universal_claims`` from the contract text itself.

  2. **composite_score()** — a single float in ``[0.0, 1.0]`` (higher = riskier)
     formed by a configurable weighted sum of risk-increasing dimensions.

  3. **triage_flags()** — a list of plain-English strings describing which
     thresholds were crossed, suitable for surfacing in a review dashboard.

  There is also **explain()** which breaks the composite score down
  dimension-by-dimension, making the number interpretable rather than opaque.

Vector layout
-------------
The numeric vector has a fixed length of 28 elements.  The layout is stable
across contracts — missing or inapplicable fields are filled with their
neutral/default value so every contract produces a vector of identical shape.

  Indices 0–4   : LLM integer scores, normalised to [0, 1]
    0  termination_risk          (1→0.0,  5→1.0)
    1  data_sensitivity          (1→0.0,  5→1.0)
    2  payment_risk              (1→0.0,  5→1.0)
    3  auto_renewal_risk         (1→0.0,  5→1.0)
    4  liability_cap_adequacy    (1→1.0,  5→0.0)  ← INVERTED: low adequacy = high risk

  Indices 5–8   : counterparty_leverage one-hot
    5  SUPPLIER   (leverage against supplier)
    6  CUSTOMER   (leverage against customer)
    7  BALANCED
    8  UNCLEAR

  Indices 9–12  : ip_ownership_risk one-hot
    9  SUPPLIER   (supplier retains IP)
    10 CUSTOMER   (IP assigned to customer)
    11 JOINT
    12 UNCLEAR

  Indices 13–17 : confidentiality_scope one-hot
    13 MUTUAL
    14 UNILATERAL_SUPPLIER   (only supplier bound)
    15 UNILATERAL_CUSTOMER   (only customer bound)
    16 NONE                  (no clause)
    17 UNCLEAR

  Indices 18–27 : deterministic signals from universal_claims
    18 cap_present                    binary (0/1)
    19 cap_formula_multiplier_value   continuous, x/(x+4) log-like transform [0,1]
    20 mutual_cap                     binary
    21 carveout_count_norm            carveouts / 3, capped at 1.0
    22 indemnity_uncapped             binary (1 = risky)
    23 mutual_indemnity               binary
    24 payment_due_days_norm          days / 90, capped at 1.0
    25 late_interest_present          binary (1 = protective; 0 = risky)
    26 auto_renewal_present           binary (1 = risk present)
    27 nonrenewal_notice_days_norm    days / 90, capped at 1.0 (higher = safer)

Usage
-----
::

    from risk_vector_normalizer import RiskVectorNormalizer

    normalizer = RiskVectorNormalizer()

    # From a saved result file
    import json
    results = json.loads(Path("results-C03.json").read_text())
    record = results[0]

    vec   = normalizer.numeric_vector(record)        # list[float], len=28
    score = normalizer.composite_score(record)       # float in [0, 1]
    flags = normalizer.triage_flags(record)          # list[str]
    exp   = normalizer.explain(record)               # dict[str, float]

    # With custom weights (override only the dimensions you care about)
    custom = RiskVectorNormalizer(weights={
        "termination_risk": 0.30,
        "data_sensitivity": 0.25,
    })
    score = custom.composite_score(record)

    # With deterministic enrichment from contract text
    text = Path("C03.txt").read_text()
    enriched = normalizer.enrich_from_text(record, text)
    vec  = normalizer.numeric_vector(enriched)       # indices 18-27 now populated

Notes
-----
- ``numeric_vector()`` and ``composite_score()`` are safe to call on a raw
  result dict even when no contract text is available — the deterministic
  fields default to their neutral values and the LLM fields do the work.
- ``enrich_from_text()`` is additive: it returns a shallow-copied record dict
  with additional ``_det_*`` keys injected.  It never mutates the input.
- When ``universal_claims`` is not importable, ``enrich_from_text()`` raises
  ``ImportError`` with a clear message.  The other three methods always work.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants — label orderings and vector layout
# ---------------------------------------------------------------------------

# These orderings must never change once records are persisted, because index
# positions in the numeric vector are stable identifiers for each dimension.
# If a new label is added to a category, append it at the end and bump the
# VECTOR_LEN constant.

_LEVERAGE_ORDER: List[str] = ["SUPPLIER", "CUSTOMER", "BALANCED", "UNCLEAR"]
_IP_ORDER: List[str]       = ["SUPPLIER", "CUSTOMER", "JOINT", "UNCLEAR"]
_SCOPE_ORDER: List[str]    = [
    "MUTUAL", "UNILATERAL_SUPPLIER", "UNILATERAL_CUSTOMER", "NONE", "UNCLEAR"
]

# Total number of elements in the numeric vector.
# 5 LLM scalars + 4 leverage + 4 ip + 5 scope + 10 deterministic = 28
VECTOR_LEN: int = 28

# Human-readable label for each vector index.  Used by explain().
VECTOR_LABELS: List[str] = [
    # LLM scalars (0-4)
    "termination_risk",
    "data_sensitivity",
    "payment_risk",
    "auto_renewal_risk",
    "liability_cap_adequacy_inv",   # inverted so higher = riskier
    # leverage one-hot (5-8)
    "leverage_SUPPLIER",
    "leverage_CUSTOMER",
    "leverage_BALANCED",
    "leverage_UNCLEAR",
    # ip one-hot (9-12)
    "ip_SUPPLIER",
    "ip_CUSTOMER",
    "ip_JOINT",
    "ip_UNCLEAR",
    # scope one-hot (13-17)
    "scope_MUTUAL",
    "scope_UNILATERAL_SUPPLIER",
    "scope_UNILATERAL_CUSTOMER",
    "scope_NONE",
    "scope_UNCLEAR",
    # deterministic (18-27)
    "cap_present",
    "cap_formula_multiplier",
    "mutual_cap",
    "carveout_count_norm",
    "indemnity_uncapped",
    "mutual_indemnity",
    "payment_due_days_norm",
    "late_interest_present",
    "auto_renewal_present",
    "nonrenewal_notice_days_norm",
]

assert len(VECTOR_LABELS) == VECTOR_LEN, (
    f"VECTOR_LABELS length {len(VECTOR_LABELS)} != VECTOR_LEN {VECTOR_LEN}"
)

# ---------------------------------------------------------------------------
# Default weights for composite_score()
# ---------------------------------------------------------------------------
# Sign convention: every dimension here contributes positively to risk.
# Dimensions with a protective connotation (late_interest, mutual_cap, etc.)
# are subtracted below in _compute_weighted_sum() rather than being given
# negative weights here, so that the weight table is easier to reason about.

_DEFAULT_WEIGHTS: Dict[str, float] = {
    # LLM dimensions — higher weight reflects greater legal significance
    "termination_risk":         0.18,
    "data_sensitivity":         0.14,
    "liability_cap_adequacy":   0.16,   # inverted inside _score_value()
    "payment_risk":             0.12,
    "auto_renewal_risk":        0.06,
    # LLM categorical — small binary penalties on top of base score
    "leverage_SUPPLIER":        0.05,   # leverage strongly against supplier
    "ip_CUSTOMER":              0.04,   # IP assigned away
    "scope_UNILATERAL_SUPPLIER":0.03,   # only supplier bound to keep secrets
    "scope_NONE":               0.04,   # no confidentiality at all
    # Deterministic — smaller weights because these are clause-level facts,
    # not holistic judgments
    "indemnity_uncapped":       0.06,
    "cap_present_inv":          0.05,   # absence of cap = risk (cap_present inverted)
    "payment_due_risk":         0.04,   # long payment terms
    "late_interest_absent":     0.03,   # no late-interest clause = risk
}

# Weights must sum to <= 1.0.  They do not need to sum to exactly 1.0;
# any remainder represents dimensions excluded from the default scoring.
_WEIGHT_SUM = sum(_DEFAULT_WEIGHTS.values())
assert _WEIGHT_SUM <= 1.001, f"Default weights sum to {_WEIGHT_SUM:.3f} (must be <= 1.0)"


# ---------------------------------------------------------------------------
# Triage threshold configuration
# ---------------------------------------------------------------------------

_TRIAGE_THRESHOLDS: Dict[str, Any] = {
    "termination_risk_high":         4,     # score >= this triggers flag
    "data_sensitivity_high":         4,
    "payment_risk_high":             4,
    "auto_renewal_risk_high":        4,
    "liability_cap_low":             2,     # score <= this triggers flag
    "composite_score_high":          0.65,  # composite score >= this triggers flag
    "payment_due_days_long":         60,    # days >= this triggers flag
}


# ---------------------------------------------------------------------------
# Helper: safe field access with fallback
# ---------------------------------------------------------------------------

def _get(record: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Return record[key] if present and not None, else default."""
    v = record.get(key)
    return default if v is None else v


# ---------------------------------------------------------------------------
# Core normalisation helpers (all pure functions, no side effects)
# ---------------------------------------------------------------------------

def _norm_score(raw: Any, default: int = 3) -> float:
    """Normalise a 1-5 integer score to [0.0, 1.0].

    ``1`` maps to ``0.0``, ``5`` maps to ``1.0``.  Values outside [1, 5] are
    clamped.  None and non-numeric values return the normalised default.
    """
    try:
        v = max(1, min(5, int(raw)))
    except (TypeError, ValueError):
        v = max(1, min(5, int(default)))
    return (v - 1) / 4.0


def _one_hot(value: Any, order: List[str], unknown_label: str = "UNCLEAR") -> List[float]:
    """Return a one-hot float list for *value* in *order*.

    Unknown or None values map to the position of *unknown_label* in *order*.
    """
    label = str(value).strip().upper() if value is not None else unknown_label
    if label not in order:
        label = unknown_label
    return [1.0 if o == label else 0.0 for o in order]


def _cap_multiplier_norm(raw: Any) -> float:
    """Map a fees-multiplier value to [0.0, 1.0] using a log-like transform.

    A multiplier of 12 (12× monthly fees — a common commercial cap) maps to
    ``12 / (12 + 4) = 0.75``.  This prevents very large multipliers from
    dominating while preserving meaningful separation at the low end.
    None or zero (no formula) maps to 0.0.
    """
    try:
        v = float(raw)
        if v <= 0:
            return 0.0
        return v / (v + 4.0)
    except (TypeError, ValueError):
        return 0.0


def _days_norm(raw: Any, cap_days: float = 90.0) -> float:
    """Normalise a day-count to [0.0, 1.0], capped at *cap_days*.

    None or zero returns 0.0.  Values above *cap_days* are clamped to 1.0.
    """
    try:
        v = float(raw)
        if v <= 0:
            return 0.0
        return min(v / cap_days, 1.0)
    except (TypeError, ValueError):
        return 0.0


def _carveout_norm(count: Any, max_carveouts: int = 3) -> float:
    """Normalise a carveout count: 0 carveouts → 0.0, ≥ max_carveouts → 1.0."""
    try:
        return min(int(count) / max_carveouts, 1.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Deterministic enrichment
# ---------------------------------------------------------------------------

def enrich_from_text(
    record: Dict[str, Any],
    contract_text: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Inject deterministic risk signals from *contract_text* into *record*.

    Calls :func:`universal_claims.extract_universal_claims` on the families
    most relevant to the risk vector (liability_cap, indemnity, payment_terms,
    auto_renewal) and adds ``_det_*`` keys to a shallow copy of *record*.

    Parameters
    ----------
    record:
        A raw result dict from ``classify_contract()``.
    contract_text:
        The full contract text.  Must be the same text that was classified.
    language:
        ``"en"`` or ``"fr"``.  When ``None``, detection is automatic.

    Returns
    -------
    A new dict (shallow copy of *record*) with ``_det_*`` keys added.
    The original *record* is never mutated.

    Raises
    ------
    ImportError
        If ``universal_claims`` is not importable.
    """
    try:
        from universal_claims import extract_universal_claims  # type: ignore
    except ImportError:
        try:
            from .universal_claims import extract_universal_claims  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "enrich_from_text() requires the 'universal_claims' module.  "
                "Ensure it is on sys.path or installed alongside this module."
            ) from exc

    enriched = dict(record)  # shallow copy — never mutate the input

    # Run the extractor on each relevant clause family in one pass over the
    # text.  We use a simple paragraph split so that each family gets the
    # most relevant paragraph rather than the whole document (which would
    # dilute signal from short but important clauses).
    paragraphs = [p.strip() for p in contract_text.split("\n\n") if p.strip()]

    # Accumulate the best (highest-confidence) projection per family.
    best: Dict[str, Dict[str, Any]] = {}
    _target_families = {
        "liability_cap", "indemnity", "payment_terms", "auto_renewal"
    }
    for para in paragraphs:
        if len(para) < 30:
            continue
        try:
            result = extract_universal_claims(
                para, language=language or "auto"
            )
        except Exception:
            continue
        fam = result.clause_family
        if fam not in _target_families:
            continue
        existing = best.get(fam)
        if existing is None or result.family_confidence > existing.get("_conf", 0.0):
            proj = dict(result.family_projection)
            proj["_conf"] = result.family_confidence
            best[fam] = proj

    # Extract the specific signals we need from each family projection.

    # ── liability_cap ─────────────────────────────────────────────────────
    cap_proj = best.get("liability_cap", {})
    enriched["_det_cap_present"]   = bool(cap_proj.get("cap_present", False))
    enriched["_det_cap_amount"]    = cap_proj.get("cap_amount")
    enriched["_det_cap_currency"]  = cap_proj.get("cap_currency")
    enriched["_det_cap_multiplier"]= cap_proj.get("cap_formula_multiplier_value")
    enriched["_det_mutual_cap"]    = bool(cap_proj.get("mutual_cap", False))
    enriched["_det_carveout_count"]= int(cap_proj.get("carveout_count", 0))

    # ── indemnity ─────────────────────────────────────────────────────────
    ind_proj = best.get("indemnity", {})
    enriched["_det_indemnity_uncapped"] = bool(ind_proj.get("indemnity_uncapped", False))
    enriched["_det_mutual_indemnity"]   = bool(ind_proj.get("mutual_indemnity", False))
    enriched["_det_indemnity_direction"]= ind_proj.get("indemnity_direction")

    # ── payment_terms ─────────────────────────────────────────────────────
    pay_proj = best.get("payment_terms", {})
    enriched["_det_payment_due_days"]   = pay_proj.get("payment_due_normalized_days")
    enriched["_det_late_interest"]      = bool(pay_proj.get("late_interest", False))
    enriched["_det_invoice_trigger"]    = bool(pay_proj.get("invoice_trigger", False))

    # ── auto_renewal ──────────────────────────────────────────────────────
    ar_proj = best.get("auto_renewal", {})
    enriched["_det_auto_renewal"]       = bool(ar_proj.get("auto_renewal", False))
    enriched["_det_nonrenewal_notice_days"] = ar_proj.get("nonrenewal_notice_normalized_days") \
        or ar_proj.get("nonrenewal_notice_value")  # fallback to raw value

    return enriched


# ---------------------------------------------------------------------------
# RiskVectorNormalizer
# ---------------------------------------------------------------------------

class RiskVectorNormalizer:
    """Converts a ``classify_contract()`` result dict into a numeric risk vector.

    Parameters
    ----------
    weights:
        Optional dict overriding specific entries in the default weight table.
        Only the keys you supply are overridden; everything else stays at its
        default.  The override is applied at construction time so each instance
        has its own immutable weight configuration.

    Examples
    --------
    ::

        n = RiskVectorNormalizer()

        vec   = n.numeric_vector(record)    # list[float], len=28
        score = n.composite_score(record)   # float in [0, 1]
        flags = n.triage_flags(record)      # list[str]
        exp   = n.explain(record)           # dict[str, float]

        # With deterministic enrichment:
        enriched = enrich_from_text(record, contract_text)
        vec = n.numeric_vector(enriched)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self._weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            for k, v in weights.items():
                self._weights[k] = float(v)

    # ── Public API ─────────────────────────────────────────────────────────

    def numeric_vector(self, record: Dict[str, Any]) -> List[float]:
        """Return the 28-element numeric risk vector for *record*.

        Layout is documented in :data:`VECTOR_LABELS`.  Every element is a
        float in [0.0, 1.0].  The vector is always exactly :data:`VECTOR_LEN`
        elements long regardless of which fields are populated.

        LLM-sourced fields (indices 0–17) require only the ``risk_vector_*``
        keys in *record*.  Deterministic fields (indices 18–27) additionally
        use ``_det_*`` keys injected by :func:`enrich_from_text`; when those
        keys are absent the deterministic slice is filled with neutral defaults.
        """
        v: List[float] = []

        # ── LLM scalars (0-4) ─────────────────────────────────────────────
        v.append(_norm_score(_get(record, "risk_vector_termination_risk",    3)))
        v.append(_norm_score(_get(record, "risk_vector_data_sensitivity",    3)))
        v.append(_norm_score(_get(record, "risk_vector_payment_risk",        3)))
        v.append(_norm_score(_get(record, "risk_vector_auto_renewal_risk",   1)))
        # liability_cap_adequacy: INVERTED — 1 (no cap) → 1.0 risk, 5 (good cap) → 0.0 risk
        raw_cap_adeq = _get(record, "risk_vector_liability_cap_adequacy", 3)
        v.append(1.0 - _norm_score(raw_cap_adeq, default=3))

        # ── leverage one-hot (5-8) ─────────────────────────────────────────
        v.extend(_one_hot(
            _get(record, "risk_vector_counterparty_leverage", "UNCLEAR"),
            _LEVERAGE_ORDER,
        ))

        # ── ip_ownership one-hot (9-12) ────────────────────────────────────
        v.extend(_one_hot(
            _get(record, "risk_vector_ip_ownership_risk", "UNCLEAR"),
            _IP_ORDER,
        ))

        # ── confidentiality_scope one-hot (13-17) ─────────────────────────
        v.extend(_one_hot(
            _get(record, "risk_vector_confidentiality_scope", "UNCLEAR"),
            _SCOPE_ORDER,
        ))

        # ── deterministic signals (18-27) ─────────────────────────────────
        # These default to neutral values when enrich_from_text has not been called.

        # 18: cap_present — protective when True, risky when absent
        v.append(1.0 if _get(record, "_det_cap_present", False) else 0.0)

        # 19: cap_formula_multiplier_value — higher multiplier = more coverage
        v.append(_cap_multiplier_norm(_get(record, "_det_cap_multiplier")))

        # 20: mutual_cap — protective binary
        v.append(1.0 if _get(record, "_det_mutual_cap", False) else 0.0)

        # 21: carveout_count — more carve-outs = better-structured cap
        v.append(_carveout_norm(_get(record, "_det_carveout_count", 0)))

        # 22: indemnity_uncapped — risky binary
        v.append(1.0 if _get(record, "_det_indemnity_uncapped", False) else 0.0)

        # 23: mutual_indemnity — balanced indemnity = protective
        v.append(1.0 if _get(record, "_det_mutual_indemnity", False) else 0.0)

        # 24: payment_due_days_norm — longer = riskier for supplier
        v.append(_days_norm(_get(record, "_det_payment_due_days"), cap_days=90.0))

        # 25: late_interest_present — protective when True
        v.append(1.0 if _get(record, "_det_late_interest", False) else 0.0)

        # 26: auto_renewal_present — risky binary
        v.append(1.0 if _get(record, "_det_auto_renewal", False) else 0.0)

        # 27: nonrenewal_notice_days_norm — longer notice = safer (protective)
        v.append(_days_norm(_get(record, "_det_nonrenewal_notice_days"), cap_days=90.0))

        assert len(v) == VECTOR_LEN, f"Vector length {len(v)} != {VECTOR_LEN}"
        return v

    def composite_score(self, record: Dict[str, Any]) -> float:
        """Return a risk score in ``[0.0, 1.0]`` (higher = riskier).

        The score is a weighted sum of risk-increasing contributions.  Each
        dimension's contribution is its normalised value multiplied by its
        weight.  Protective signals (cap present, late interest) subtract
        from the score rather than adding to it.

        Returns a value rounded to 4 decimal places.
        """
        w = self._weights
        vec = self.numeric_vector(record)

        score = 0.0

        # ── LLM scalars ────────────────────────────────────────────────────
        score += vec[0] * w.get("termination_risk",       0.0)
        score += vec[1] * w.get("data_sensitivity",       0.0)
        score += vec[2] * w.get("payment_risk",           0.0)
        score += vec[3] * w.get("auto_renewal_risk",      0.0)
        score += vec[4] * w.get("liability_cap_adequacy", 0.0)  # already inverted in vec

        # ── LLM categorical ────────────────────────────────────────────────
        # leverage: supplier-disadvantaged leverage adds to risk score
        score += vec[5]  * w.get("leverage_SUPPLIER",         0.0)   # idx 5 = SUPPLIER
        # ip: IP assigned to customer adds risk for supplier
        score += vec[10] * w.get("ip_CUSTOMER",               0.0)   # idx 10 = CUSTOMER
        # scope: unilateral (supplier only) or absent conf adds risk
        score += vec[14] * w.get("scope_UNILATERAL_SUPPLIER", 0.0)   # idx 14
        score += vec[16] * w.get("scope_NONE",                0.0)   # idx 16

        # ── Deterministic ──────────────────────────────────────────────────
        # Absence of a cap is risky: use (1 - cap_present)
        score += (1.0 - vec[18]) * w.get("cap_present_inv",    0.0)
        # Uncapped indemnity is risky
        score += vec[22]         * w.get("indemnity_uncapped",  0.0)
        # Long payment terms: use payment_due_days_norm directly
        score += vec[24]         * w.get("payment_due_risk",    0.0)
        # Absence of late interest clause is risky: use (1 - late_interest)
        score += (1.0 - vec[25]) * w.get("late_interest_absent", 0.0)

        return round(min(max(score, 0.0), 1.0), 4)

    def triage_flags(self, record: Dict[str, Any]) -> List[str]:
        """Return a list of plain-English risk flags for *record*.

        Each string describes one threshold that was crossed.  An empty list
        means no configured threshold was triggered — not necessarily "safe",
        but no automated flag was raised.

        Flags are ordered roughly from most to least severe.
        """
        flags: List[str] = []
        t = _TRIAGE_THRESHOLDS

        # ── LLM score thresholds ───────────────────────────────────────────
        term = _get(record, "risk_vector_termination_risk", 3)
        if isinstance(term, (int, float)) and term >= t["termination_risk_high"]:
            flags.append(
                f"HIGH termination risk (score {term}/5): unilateral or short-notice "
                "termination right present."
            )

        cap_adeq = _get(record, "risk_vector_liability_cap_adequacy", 3)
        if isinstance(cap_adeq, (int, float)) and cap_adeq <= t["liability_cap_low"]:
            flags.append(
                f"Liability cap inadequate or absent (score {cap_adeq}/5): "
                "exposure may be disproportionate to contract value."
            )

        data = _get(record, "risk_vector_data_sensitivity", 3)
        if isinstance(data, (int, float)) and data >= t["data_sensitivity_high"]:
            flags.append(
                f"HIGH data sensitivity (score {data}/5): special-category or "
                "sensitive personal data processed — GDPR Art. 9 / Art. 28 review required."
            )

        pay = _get(record, "risk_vector_payment_risk", 3)
        if isinstance(pay, (int, float)) and pay >= t["payment_risk_high"]:
            flags.append(
                f"HIGH payment risk (score {pay}/5): vague payment triggers, "
                "long terms, or no late-payment remedy."
            )

        renewal = _get(record, "risk_vector_auto_renewal_risk", 1)
        if isinstance(renewal, (int, float)) and renewal >= t["auto_renewal_risk_high"]:
            flags.append(
                f"Auto-renewal trap (score {renewal}/5): opt-out window is very "
                "short or unclear — calendar management required."
            )

        # ── LLM categorical checks ─────────────────────────────────────────
        leverage = _get(record, "risk_vector_counterparty_leverage", "UNCLEAR")
        if leverage == "SUPPLIER":
            flags.append(
                "Counterparty leverage: SUPPLIER — obligations and risk fall "
                "disproportionately on the supplying party."
            )

        ip = _get(record, "risk_vector_ip_ownership_risk", "UNCLEAR")
        if ip == "CUSTOMER":
            flags.append(
                "IP ownership: assigned to CUSTOMER — verify scope of the "
                "assignment clause; pre-existing IP and background IP carve-outs."
            )

        scope = _get(record, "risk_vector_confidentiality_scope", "UNCLEAR")
        if scope == "NONE":
            flags.append(
                "Confidentiality: NO clause detected — confidential information "
                "exchanged under this contract is unprotected."
            )
        elif scope == "UNILATERAL_SUPPLIER":
            flags.append(
                "Confidentiality: UNILATERAL (supplier only) — only the supplier "
                "is bound; customer's obligations to keep information confidential "
                "are absent or unclear."
            )

        # ── Deterministic checks (only fire when enrichment has been run) ──
        if _get(record, "_det_indemnity_uncapped", False):
            flags.append(
                "Indemnity: UNCAPPED — indemnification obligation has no stated "
                "monetary ceiling; exposure is potentially unlimited."
            )

        det_pay_days = _get(record, "_det_payment_due_days")
        if det_pay_days is not None:
            try:
                if float(det_pay_days) >= t["payment_due_days_long"]:
                    flags.append(
                        f"Payment terms: {int(det_pay_days)} days — longer than the "
                        f"{t['payment_due_days_long']}-day threshold."
                    )
            except (TypeError, ValueError):
                pass

        if _get(record, "_det_auto_renewal", False) and not _get(record, "_det_nonrenewal_notice_days"):
            flags.append(
                "Auto-renewal: clause detected but no opt-out notice period found — "
                "renewal may be unconditional."
            )

        # ── Composite score threshold ──────────────────────────────────────
        score = self.composite_score(record)
        if score >= t["composite_score_high"]:
            flags.append(
                f"Composite risk score {score:.2f} ≥ {t['composite_score_high']} "
                "threshold — overall risk profile is elevated."
            )

        return flags

    def explain(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Return a per-dimension breakdown of the composite score contribution.

        Each key is a dimension name; each value is the contribution of that
        dimension to the composite score (weight × normalised value).  The sum
        of all values equals ``composite_score(record)``.

        Useful for understanding why a contract scored the way it did.

        Example output::

            {
                "termination_risk":       0.1350,   # scored 4/5 × weight 0.18
                "data_sensitivity":       0.0350,   # scored 2/5 × weight 0.14
                "liability_cap_adequacy": 0.1200,   # inverted 2/5=0.75 × weight 0.16
                "payment_risk":           0.0600,   # scored 3/5 × weight 0.12
                "auto_renewal_risk":      0.0000,   # scored 1/5=0.0 × weight 0.06
                "leverage_SUPPLIER":      0.0000,   # one-hot 0 × weight 0.05
                ...
                "_composite_score":       0.3944,   # sum of all contributions
            }
        """
        w = self._weights
        vec = self.numeric_vector(record)

        parts: Dict[str, float] = {}

        parts["termination_risk"]          = round(vec[0]         * w.get("termination_risk",       0.0), 6)
        parts["data_sensitivity"]          = round(vec[1]         * w.get("data_sensitivity",       0.0), 6)
        parts["payment_risk"]              = round(vec[2]         * w.get("payment_risk",           0.0), 6)
        parts["auto_renewal_risk"]         = round(vec[3]         * w.get("auto_renewal_risk",      0.0), 6)
        parts["liability_cap_adequacy_inv"]= round(vec[4]         * w.get("liability_cap_adequacy", 0.0), 6)
        parts["leverage_SUPPLIER"]         = round(vec[5]         * w.get("leverage_SUPPLIER",         0.0), 6)
        parts["ip_CUSTOMER"]               = round(vec[10]        * w.get("ip_CUSTOMER",               0.0), 6)
        parts["scope_UNILATERAL_SUPPLIER"] = round(vec[14]        * w.get("scope_UNILATERAL_SUPPLIER", 0.0), 6)
        parts["scope_NONE"]                = round(vec[16]        * w.get("scope_NONE",                0.0), 6)
        parts["cap_absent"]                = round((1.0 - vec[18])* w.get("cap_present_inv",    0.0), 6)
        parts["indemnity_uncapped"]        = round(vec[22]        * w.get("indemnity_uncapped",  0.0), 6)
        parts["payment_due_risk"]          = round(vec[24]        * w.get("payment_due_risk",    0.0), 6)
        parts["late_interest_absent"]      = round((1.0 - vec[25])* w.get("late_interest_absent",0.0), 6)

        parts["_composite_score"] = round(sum(v for k, v in parts.items() if not k.startswith("_")), 4)
        return parts
