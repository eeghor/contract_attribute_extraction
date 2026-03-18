"""
Contract risk vector assembler.

Architecture
------------
Two-pass computation:

Pass 1 (deterministic) — extracts dimensions that are fully recoverable
from pattern matching.  No LLM.  Calls universal_claims.extract_universal_claims()
on each clause in the pre-segmented contract.  Covers ~60% of the vector.

Pass 2 (LLM gap-fill, once at ingestion) — calls contract_classifier.py's
OpenRouter infrastructure with a new RiskVectorPrompt for the dimensions
requiring legal judgment: normalized scores, cross-clause reasoning, and
detection of meaningful absences.

Output: RiskVector dataclass — a fixed-width vector with a parallel
confidence array indicating which dimensions came from deterministic
extraction vs LLM inference.

Perspective: vectors are neutral feature representations.
Risk interpretation (which party is exposed) is a separate weighted
scoring function applied on top, parameterized by party perspective.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import math

@dataclass
class RiskVector:
    # ── Tier A: fully deterministic dims ─────────────────────────────────
    # All extracted from universal_claims family_projection fields.
    # Confidence: HIGH when family_confidence > 0.7, MEDIUM otherwise.

    # Liability cap
    cap_present: bool = False
    cap_amount_raw: Optional[float] = None          # absolute value in contract currency
    cap_currency: Optional[str] = None
    cap_is_fees_multiple: bool = False
    cap_multiplier: Optional[float] = None          # e.g. 1.0, 2.0, 0.5
    cap_mutual: bool = False
    cap_carveout_fraud: bool = False
    cap_carveout_gross_negligence: bool = False
    cap_carveout_wilful_misconduct: bool = False
    cap_carveout_count: int = 0                     # number of carve-outs (key signal)
    consequential_loss_excluded: bool = False

    # Termination
    termination_for_convenience: bool = False
    convenience_notice_days: Optional[float] = None
    termination_for_breach: bool = False
    cure_period_days: Optional[float] = None
    insolvency_trigger: bool = False

    # Payment
    payment_days: Optional[float] = None            # normalized to days
    late_interest: bool = False

    # IP
    ip_ownership_assigned: bool = False             # True = work-for-hire / assignment
    ip_licence_exclusive: bool = False
    ip_sublicence_permitted: bool = False
    ip_perpetual: bool = False

    # Auto-renewal
    auto_renewal: bool = False
    renewal_term_months: Optional[float] = None
    non_renewal_notice_days: Optional[float] = None
    fr_mandatory_notice_risk: bool = False          # French L.215-1 advisory

    # Data / GDPR
    dpa_present: bool = False
    cross_border_transfer: bool = False
    retention_period_days: Optional[float] = None

    # Dispute resolution
    dispute_mechanism: Optional[str] = None         # "arbitration" | "mediation" | "litigation"
    arbitration_institution: Optional[str] = None   # "icc" | "lcia" | "cmap" etc.
    governing_law: Optional[str] = None             # raw string, normalized separately
    governing_law_category: Optional[str] = None    # "french" | "english" | "us_ny" | "other"

    # Agreement duration
    agreement_term_months: Optional[float] = None
    indefinite_term: bool = False

    # Warranty
    warranty_given: bool = False
    as_is_disclaimer: bool = False
    warranty_period_days: Optional[float] = None

    # ── Tier B: LLM gap-fill dims ─────────────────────────────────────────
    # Set to None until LLM call completes.  Caller must check None before use.

    # Normalized risk scores (0.0–1.0, vendor perspective: 1.0 = most protective for vendor)
    cap_risk_score: Optional[float] = None          # 0=uncapped/high risk, 1=low cap/low risk
    indemnity_risk_score: Optional[float] = None    # 0=broad unilateral, 1=mutual/narrow
    ip_risk_score: Optional[float] = None           # 0=full assignment to client, 1=vendor retains
    overall_balance_score: Optional[float] = None   # 0=heavily client-favorable, 1=vendor-favorable

    # Absence detection (LLM only — deterministic cannot detect meaningful absence)
    liability_cap_absent: bool = False              # no cap clause found and none implied
    indemnity_uncapped_risk: bool = False           # indemnity present but cap carve-out applies

    # New materiality flags (post-2023)
    ai_training_permitted: Optional[bool] = None
    broad_audit_rights: Optional[bool] = None

    # ── Metadata ──────────────────────────────────────────────────────────
    confidence: Dict[str, str] = field(default_factory=dict)
    # key: dimension name, value: "HIGH" | "MEDIUM" | "LLM" | "ABSENT"
    source_clauses: Dict[str, str] = field(default_factory=dict)
    # key: dimension name, value: clause family that populated it


def assemble_risk_vector(
    clause_inventory: list,        # list of ClaimExtractionResult from universal_claims
    contract_value: Optional[float] = None,   # for normalizing cap amounts
) -> RiskVector:
    """
    Pass 1: deterministic assembly from clause inventory.
    Returns a partially populated RiskVector with Tier A dims filled.
    Tier B dims remain None until enrich_with_llm() is called.
    """
    rv = RiskVector()

    for result in clause_inventory:
        proj = result.family_projection
        conf = "HIGH" if result.family_confidence > 0.7 else "MEDIUM"
        family = result.clause_family

        if family == "liability_cap":
            rv.cap_present = True
            rv.cap_amount_raw = proj.get("cap_amount")
            rv.cap_currency = proj.get("cap_currency")
            rv.cap_is_fees_multiple = proj.get("cap_formula_fees_paid", False)
            rv.cap_multiplier = proj.get("cap_formula_multiplier_value")  # new field
            rv.cap_mutual = proj.get("mutual_cap", False)
            rv.cap_carveout_fraud = proj.get("carveout_fraud", False)
            rv.cap_carveout_gross_negligence = proj.get("carveout_gross_negligence", False)
            rv.cap_carveout_wilful_misconduct = proj.get("carveout_wilful_misconduct", False)
            rv.cap_carveout_count = sum([
                rv.cap_carveout_fraud,
                rv.cap_carveout_gross_negligence,
                rv.cap_carveout_wilful_misconduct,
            ])
            rv.consequential_loss_excluded = proj.get("excludes_indirect_damages", False)
            rv.confidence["cap"] = conf
            rv.source_clauses["cap"] = family

        elif family == "termination_notice":
            rv.termination_for_convenience = proj.get("termination_for_convenience", False)
            rv.convenience_notice_days = proj.get("notice_period_normalized_days")
            rv.termination_for_breach = proj.get("termination_for_breach", False)
            rv.cure_period_days = proj.get("cure_period_normalized_days")
            rv.insolvency_trigger = proj.get("insolvency_trigger", False)
            rv.confidence["termination"] = conf

        elif family == "payment_terms":
            rv.payment_days = proj.get("payment_due_normalized_days")
            rv.late_interest = proj.get("late_interest", False)
            rv.confidence["payment"] = conf

        elif family == "ip_licence":
            rv.ip_ownership_assigned = proj.get("ip_ownership_assigned", False)
            rv.ip_licence_exclusive = proj.get("exclusive", False)
            rv.ip_sublicence_permitted = proj.get("sublicence_permitted", False)
            rv.ip_perpetual = proj.get("perpetual_licence", False)
            rv.confidence["ip"] = conf

        elif family == "auto_renewal":
            rv.auto_renewal = proj.get("auto_renewal", False)
            rv.renewal_term_months = proj.get("renewal_term_normalized_months")
            rv.non_renewal_notice_days = proj.get("nonrenewal_notice_normalized_days")
            rv.fr_mandatory_notice_risk = proj.get("fr_mandatory_notice_risk", False)
            rv.confidence["auto_renewal"] = conf

        elif family == "data_retention":
            rv.dpa_present = proj.get("dpa_present", False)
            rv.cross_border_transfer = proj.get("cross_border_transfer_mentioned", False)
            rv.retention_period_days = proj.get("retention_period_normalized_days")
            rv.confidence["data"] = conf

        elif family == "dispute_resolution":
            rv.dispute_mechanism = proj.get("dispute_resolution_type")
            rv.arbitration_institution = proj.get("arbitration_institution")
            rv.confidence["dispute"] = conf

        elif family == "governing_law":
            rv.governing_law = proj.get("governing_law")
            rv.governing_law_category = _categorize_governing_law(rv.governing_law)
            rv.confidence["governing_law"] = conf

        elif family == "agreement_term":
            rv.agreement_term_months = proj.get("initial_term_normalized_months")
            rv.indefinite_term = proj.get("indefinite_term", False)
            rv.confidence["term"] = conf

        elif family == "warranty":
            rv.warranty_given = proj.get("warranty_given", False)
            rv.as_is_disclaimer = proj.get("disclaimer_present", False)
            rv.warranty_period_days = proj.get("warranty_period_normalized_days")
            rv.confidence["warranty"] = conf

    # Absence detection (what deterministic CAN catch: simply not populated)
    if not rv.cap_present:
        rv.liability_cap_absent = True
        rv.confidence["cap"] = "ABSENT"

    return rv


def _categorize_governing_law(raw: Optional[str]) -> Optional[str]:
    """Map raw governing law string to canonical category for one-hot encoding."""
    if not raw:
        return None
    lower = raw.lower()
    if any(k in lower for k in ("french", "france", "droit français", "droit francais")):
        return "french"
    if any(k in lower for k in ("english", "england", "laws of england")):
        return "english"
    if any(k in lower for k in ("new york", "delaware", "laws of the state")):
        return "us_ny"
    if "german" in lower or "deutsches" in lower:
        return "german"
    if "dutch" in lower or "netherlands" in lower:
        return "dutch"
    return "other"


def to_similarity_vector(rv: RiskVector, contract_value: Optional[float] = None) -> Dict[str, Any]:
    """
    Convert a RiskVector to a flat dict suitable for cosine similarity computation.

    Numerical dims are normalized to [0, 1].
    Categoricals are one-hot encoded.
    Missing values (None) are filled with the population median for that dimension,
    flagged in the parallel confidence dict so callers can discount imputed values.

    Note: governing_law is NOT included in this vector — it should be used as a
    pre-filter before similarity search, not as a distance dimension.
    """
    vec = {}

    # Cap dimensions
    vec["cap_present"] = 1.0 if rv.cap_present else 0.0
    # Normalize cap amount: if fees multiple, use multiplier directly; else use log-scale
    # relative to contract value if available
    if rv.cap_is_fees_multiple and rv.cap_multiplier is not None:
        # 1.0x fees → 0.5, 2.0x → 0.67, 0.5x → 0.33, uncapped → 0.0
        vec["cap_level"] = rv.cap_multiplier / (rv.cap_multiplier + 1.0)
    elif rv.cap_amount_raw is not None and contract_value and contract_value > 0:
        ratio = rv.cap_amount_raw / contract_value
        vec["cap_level"] = min(ratio / (ratio + 1.0), 1.0)  # log-like normalization
    else:
        vec["cap_level"] = 0.5  # unknown — imputed median
    vec["cap_mutual"] = 1.0 if rv.cap_mutual else 0.0
    vec["carveout_count_norm"] = min(rv.cap_carveout_count / 3.0, 1.0)
    vec["consequential_excluded"] = 1.0 if rv.consequential_loss_excluded else 0.0

    # Termination
    vec["termination_convenience"] = 1.0 if rv.termination_for_convenience else 0.0
    # Notice period: 30d→0.3, 90d→0.6, 180d→0.8, >365d→1.0
    if rv.convenience_notice_days is not None:
        vec["notice_period_norm"] = min(rv.convenience_notice_days / 365.0, 1.0)
    else:
        vec["notice_period_norm"] = 0.5
    vec["insolvency_trigger"] = 1.0 if rv.insolvency_trigger else 0.0

    # IP
    vec["ip_assigned"] = 1.0 if rv.ip_ownership_assigned else 0.0
    vec["ip_exclusive"] = 1.0 if rv.ip_licence_exclusive else 0.0
    vec["ip_perpetual"] = 1.0 if rv.ip_perpetual else 0.0

    # Auto-renewal risk
    vec["auto_renewal"] = 1.0 if rv.auto_renewal else 0.0
    if rv.non_renewal_notice_days is not None:
        vec["renewal_notice_norm"] = min(rv.non_renewal_notice_days / 90.0, 1.0)
    else:
        vec["renewal_notice_norm"] = 0.5
    vec["fr_notice_risk"] = 1.0 if rv.fr_mandatory_notice_risk else 0.0

    # Data / GDPR
    vec["dpa_present"] = 1.0 if rv.dpa_present else 0.0
    vec["cross_border"] = 1.0 if rv.cross_border_transfer else 0.0

    # Warranty
    vec["warranty_given"] = 1.0 if rv.warranty_given else 0.0
    vec["as_is_disclaimer"] = 1.0 if rv.as_is_disclaimer else 0.0

    # Dispute mechanism — one-hot (governing_law excluded, used as pre-filter)
    vec["dispute_arbitration"] = 1.0 if rv.dispute_mechanism == "arbitration" else 0.0
    vec["dispute_litigation"] = 1.0 if rv.dispute_mechanism == "litigation" else 0.0

    # LLM scores (only included when available; similarity queries can exclude these dims)
    if rv.cap_risk_score is not None:
        vec["cap_risk_score"] = rv.cap_risk_score
    if rv.indemnity_risk_score is not None:
        vec["indemnity_risk_score"] = rv.indemnity_risk_score
    if rv.overall_balance_score is not None:
        vec["overall_balance"] = rv.overall_balance_score

    return vec