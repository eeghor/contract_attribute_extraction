#!/usr/bin/env python3
"""Contract information extractor with a "cheap checks first" workflow.

Plain-English overview
----------------------
Think of this script as a two-step review process:

1. It first looks for obvious facts using fixed rules.
   Example: if the text literally says "Effective Date: 1 January 2025",
   we do not need an AI model to understand that.

2. It asks an LLM only for the parts that are still unclear.
   Example: deciding which party is the internal entity can require more
   judgment than a simple pattern match.

Why this approach exists
------------------------
- It is faster because many fields can be read directly from the text.
- It is cheaper because the LLM is only used when needed.
- It is safer because obvious facts are handled by deterministic rules,
  which are more predictable than a model.
- It is optimized to avoid repeated work: the contract is normalized once,
  regex-heavy checks reuse that prepared text, and the LLM sees a small
  field-specific schema instead of the full output schema.

High-level flow
---------------
1. Read the contract text.
2. Preprocess the contract once into reusable lines and clauses.
3. Run local regex extractors for the fields that are usually explicit.
4. Send the "sometimes present" fields directly to the LLM.
5. If any regex-first field is still empty, also send it to the LLM.
6. Ignore fields that are usually internal metadata and are not worth
   searching for in the contract text.
7. Save any high-confidence values together with evidence quotes.
8. Return one JSON object with extracted values, evidence, warnings, and the
   elapsed time in seconds.

Important behavior
------------------
- Local matches win when they are present. The LLM does not overwrite them.
- If the LLM is unavailable, the script still returns partial local results.
- Every non-null field should carry evidence showing where it came from.
- The LLM request is intentionally small: only unresolved or LLM-only fields
  are included, together with a matching mini-schema.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import Enum
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


class PrimaryContractType(str, Enum):
    MODIFICATION_AND_CLOSURE = "MODIFICATION_AND_CLOSURE"
    CORPORATE_AND_STRUCTURAL = "CORPORATE_AND_STRUCTURAL"
    FINANCE_AND_TREASURY = "FINANCE_AND_TREASURY"
    INDIVIDUAL_LABOUR = "INDIVIDUAL_LABOUR"
    REAL_ESTATE_AND_FACILITIES = "REAL_ESTATE_AND_FACILITIES"
    IP_LICENSING_AND_TECH = "IP_LICENSING_AND_TECH"
    SUPPLY_AND_TRADE = "SUPPLY_AND_TRADE"
    SERVICES = "SERVICES"
    LEGAL_SETTLEMENT_AND_RIGHTS = "LEGAL_SETTLEMENT_AND_RIGHTS"


class PaymentPeriodicity(str, Enum):
    ONE_OFF = "ONE_OFF"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"
    MILESTONE_BASED = "MILESTONE_BASED"
    USAGE_BASED = "USAGE_BASED"
    ON_DELIVERY = "ON_DELIVERY"
    OTHER = "OTHER"


class Language(str, Enum):
    EN = "en"
    FR = "fr"
    MIXED = "mixed"


class EvidenceItem(BaseModel):
    """Evidence attached to one extracted field."""

    model_config = ConfigDict(extra="forbid")

    quote: Optional[str] = None
    note: Optional[str] = None
    derived: bool = False


class Evidence(BaseModel):
    """Evidence container for all extractable fields."""

    model_config = ConfigDict(extra="forbid")

    primary_contract_type: EvidenceItem
    internal_entity: EvidenceItem
    business_unit: EvidenceItem
    sales_owner: EvidenceItem
    legal_owner: EvidenceItem
    counterparty_name: EvidenceItem
    counterparty_contact_email: EvidenceItem
    counterparty_signatory: EvidenceItem
    signature_date: EvidenceItem
    effective_date: EvidenceItem
    start_date: EvidenceItem
    end_date: EvidenceItem
    auto_renewal: EvidenceItem
    renewal_notice_days: EvidenceItem
    termination_notice_days: EvidenceItem
    contract_term: EvidenceItem
    total_contract_value: EvidenceItem
    currency: EvidenceItem
    payment_periodicity: EvidenceItem
    payment_terms_days: EvidenceItem
    indexation: EvidenceItem


class ContractExtraction(BaseModel):
    """Final JSON payload returned by the extractor."""

    model_config = ConfigDict(extra="forbid")

    language: Optional[Language] = None
    primary_contract_type: Optional[PrimaryContractType] = None
    internal_entity: Optional[str] = None
    business_unit: Optional[str] = None
    sales_owner: Optional[str] = None
    legal_owner: Optional[str] = None
    counterparty_name: Optional[str] = None
    counterparty_contact_email: Optional[str] = None
    counterparty_signatory: Optional[str] = None
    signature_date: Optional[str] = None
    effective_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    auto_renewal: Optional[bool] = None
    renewal_notice_days: Optional[int] = None
    termination_notice_days: Optional[int] = None
    contract_term: Optional[str] = None
    total_contract_value: Optional[float] = None
    currency: Optional[str] = None
    payment_periodicity: Optional[PaymentPeriodicity] = None
    payment_terms_days: Optional[int] = None
    indexation: Optional[str] = None
    evidence: Evidence
    warnings: list[str] = Field(default_factory=list)
    elapsed_time_seconds: int = 0


@dataclass
class Candidate:
    """Internal representation of one possible field value before final merge."""

    value: Any = None
    quote: Optional[str] = None
    note: Optional[str] = None
    derived: bool = False
    confidence: float = 0.0
    source: str = "regex"


@dataclass
class PreparedContract:
    """Normalized contract view reused across regex extractors.

    The main speed benefit is that expensive text preparation happens once,
    instead of every extractor lowercasing and splitting the same contract
    repeatedly.
    """

    text: str
    lines: list[str]
    folded_lines: list[str]
    clauses: list[str]
    folded_clauses: list[str]
    header_text: str
    header_lines: list[str]
    folded_header: str


EVIDENCE_FIELDS = [
    "primary_contract_type",
    "internal_entity",
    "business_unit",
    "sales_owner",
    "legal_owner",
    "counterparty_name",
    "counterparty_contact_email",
    "counterparty_signatory",
    "signature_date",
    "effective_date",
    "start_date",
    "end_date",
    "auto_renewal",
    "renewal_notice_days",
    "termination_notice_days",
    "contract_term",
    "total_contract_value",
    "currency",
    "payment_periodicity",
    "payment_terms_days",
    "indexation",
]

ALL_FIELDS = ["language", *EVIDENCE_FIELDS]
REGEX_FIRST_FIELDS = [
    "primary_contract_type",
    "counterparty_name",
    "counterparty_signatory",
    "signature_date",
    "effective_date",
    "start_date",
    "end_date",
    "auto_renewal",
    "renewal_notice_days",
    "termination_notice_days",
    "contract_term",
    "currency",
    "payment_periodicity",
    "payment_terms_days",
]
LLM_ONLY_FIELDS = [
    "internal_entity",
    "counterparty_contact_email",
    "total_contract_value",
    "indexation",
]
EXCLUDED_FROM_SEARCH_FIELDS = [
    "language",
    "business_unit",
    "sales_owner",
    "legal_owner",
]
ROUTED_FIELDS = [*REGEX_FIRST_FIELDS, *LLM_ONLY_FIELDS]

DATE_VALUE_PATTERN = (
    r"(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{4}|"
    r"\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4}|"
    r"[A-Za-zÀ-ÿ]+\s+\d{1,2},?\s+\d{4})"
)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PAYMENT_CLAUSE_CUES = ("fee", "fees", "price", "pricing", "pay", "payable", "invoice", "billing", "charge", "subscription", "rent", "loyer", "factur", "paiement", "redevance", "tarif")
EXTERNAL_EMAIL_CUES = (
    "counterparty",
    "customer",
    "client",
    "vendor",
    "supplier",
    "provider",
    "contractor",
    "consultant",
    "licensor",
    "licensee",
    "prestataire",
    "fournisseur",
    "cocontractant",
    "external",
)
INTERNAL_EMAIL_CUES = (
    "company",
    "our",
    "internal",
    "legal",
    "finance",
    "accounts payable",
    "comptabilite",
    "comptabilité",
    "juridique",
)
INDEXATION_CUES = (
    "cpi",
    "ipc",
    "ilc",
    "ilat",
    "icc",
    "syntec",
    "inflation",
    "indexation",
    "revision",
    "révision",
    "increase",
    "adjust",
    "uplift",
)
LOCK_CONFIDENCE_BY_FIELD: dict[str, float] = {
    "primary_contract_type": 0.85,
    "counterparty_name": 0.92,
    "counterparty_signatory": 0.92,
    "signature_date": 0.95,
    "effective_date": 0.95,
    "start_date": 0.95,
    "end_date": 0.95,
    "auto_renewal": 0.95,
    "renewal_notice_days": 0.9,
    "termination_notice_days": 0.9,
    "contract_term": 0.9,
    "currency": 0.9,
    "payment_periodicity": 0.9,
    "payment_terms_days": 0.95,
}
PRIMARY_TYPE_RULES: list[tuple[PrimaryContractType, list[str]]] = [
    (
        PrimaryContractType.MODIFICATION_AND_CLOSURE,
        ["amendment", "addendum", "change order", "extension", "termination", "termination notice", "waiver", "avenant", "résiliation", "resiliation", "prorogation", "renewal letter"],
    ),
    (
        PrimaryContractType.CORPORATE_AND_STRUCTURAL,
        ["shareholders", "articles of association", "board resolution", "share purchase agreement", "stock purchase agreement", "corporate resolution"],
    ),
    (
        PrimaryContractType.FINANCE_AND_TREASURY,
        ["loan agreement", "facility agreement", "credit agreement", "guarantee", "promissory note", "intercompany loan"],
    ),
    (
        PrimaryContractType.INDIVIDUAL_LABOUR,
        ["employment agreement", "employment contract", "contrat de travail", "salary", "paid leave", "lien de subordination", "employee", "salarié"],
    ),
    (
        PrimaryContractType.REAL_ESTATE_AND_FACILITIES,
        ["lease agreement", "commercial lease", "bail commercial", "lease", "bail", "premises", "locaux", "construction contract", "epc"],
    ),
    (
        PrimaryContractType.IP_LICENSING_AND_TECH,
        ["saas", "software license", "licence", "license agreement", "platform access", "logiciel", "subscription services"],
    ),
    (
        PrimaryContractType.SUPPLY_AND_TRADE,
        ["supply agreement", "purchase order", "sale of goods", "logistics", "distribution agreement", "fourniture", "vente", "livraison"],
    ),
    (
        PrimaryContractType.SERVICES,
        ["services agreement", "master services agreement", "statement of work", "consulting", "professional services", "prestations de services", "assistance", "conseil"],
    ),
    (
        PrimaryContractType.LEGAL_SETTLEMENT_AND_RIGHTS,
        ["nda", "non-disclosure", "confidentiality agreement", "settlement agreement", "release", "power of attorney"],
    ),
]
LLM_FIELD_RULES: dict[str, str] = {
    "language": "Set to en, fr, mixed, or null based only on the provided snippets.",
    "primary_contract_type": "Choose exactly one taxonomy value or null. Modification/termination documents always map to MODIFICATION_AND_CLOSURE.",
    "internal_entity": "Return the internal signing entity only if clearly identified in party definitions or signature blocks.",
    "business_unit": "Return only if an internal business unit or division is explicitly named.",
    "sales_owner": "Return only if an internal commercial owner is explicitly named.",
    "legal_owner": "Return only if an internal legal owner or counsel is explicitly named.",
    "counterparty_name": "Return the external legal contracting party only if explicit.",
    "counterparty_signatory": "Return the external signatory only if explicit.",
    "counterparty_contact_email": "Return the external counterparty contact email only if explicit.",
    "signature_date": "Return the signature or execution date normalized to YYYY-MM-DD.",
    "effective_date": "Return the effective date normalized to YYYY-MM-DD.",
    "start_date": "Return the contract or service start date normalized to YYYY-MM-DD.",
    "end_date": "Return an explicit end date normalized to YYYY-MM-DD. Derive only if explicit base date plus fixed term are both stated.",
    "auto_renewal": "Return true only for explicit automatic or tacit renewal, false only for explicit no-auto-renewal language, else null.",
    "renewal_notice_days": "Return the day count needed to prevent renewal only if explicit in days.",
    "termination_notice_days": "Return the ordinary termination notice period in days only if explicit.",
    "contract_term": "Normalize concise text such as '12 months from Effective Date', 'until 2028-12-31', or 'indefinite'.",
    "total_contract_value": "Return only a binding committed total amount. Do not infer from incomplete pricing.",
    "currency": "Normalize to an ISO 4217 code only if unambiguous.",
    "payment_periodicity": "Return one enum value only if the billing cadence is explicit.",
    "payment_terms_days": "Return invoice payment days such as Net 30 or payable within 45 days.",
    "indexation": "Return a short normalized description of any explicit indexation or annual uplift mechanism, else null.",
}
FIELD_KEYWORDS: dict[str, list[str]] = {
    "language": ["whereas", "attendu", "agreement", "contrat"],
    "primary_contract_type": ["amendment", "addendum", "agreement", "contract", "avenant", "termination", "license", "services", "lease", "loan"],
    "internal_entity": ["between", "among", "party", "parties", "signature", "signed by", "for and on behalf of"],
    "business_unit": ["business unit", "division", "department"],
    "sales_owner": ["sales owner", "account owner", "commercial owner"],
    "legal_owner": ["legal owner", "legal contact", "counsel", "juridique"],
    "counterparty_name": ["between", "among", "party", "parties", "signature", "signed by"],
    "counterparty_contact_email": ["email", "e-mail", "courriel", "notice"],
    "counterparty_signatory": ["signature", "signed by", "for and on behalf of", "name:", "titre"],
    "signature_date": ["signature", "executed", "signed on", "dated"],
    "effective_date": ["effective date", "date d'effet", "prise d'effet", "entry into force", "entrée en vigueur"],
    "start_date": ["start date", "commencement", "date de début", "service commencement", "debut"],
    "end_date": ["end date", "expiration", "expiry", "date de fin", "échéance", "echeance"],
    "auto_renewal": ["automatic renewal", "tacite reconduction", "renew automatically", "renouvellement automatique"],
    "renewal_notice_days": ["renewal", "renew", "reconduction", "renouvellement", "notice", "préavis", "preavis"],
    "termination_notice_days": ["termination", "terminate", "resiliation", "résiliation", "notice", "préavis", "preavis"],
    "contract_term": ["term", "duration", "durée", "duree", "initial term"],
    "total_contract_value": ["total fees", "total amount", "contract value", "montant total", "prix total", "minimum commitment"],
    "currency": ["eur", "usd", "gbp", "chf", "cad", "€", "$", "£"],
    "payment_periodicity": ["monthly", "quarterly", "annually", "monthly fee", "facture", "mensuel", "trimestriel"],
    "payment_terms_days": ["net 30", "payable within", "within", "invoice", "paiement", "facture"],
    "indexation": ["cpi", "ipc", "ilc", "ilat", "icc", "syntec", "inflation", "indexation", "révision", "revision"],
}
PRIMARY_TYPE_TAXONOMY = """
MODIFICATION_AND_CLOSURE: amendments, addenda, extensions, waivers, renewals, terminations.
CORPORATE_AND_STRUCTURAL: shareholders agreements, articles, SPAs, board resolutions.
FINANCE_AND_TREASURY: loans, facilities, guarantees, intercompany debt.
INDIVIDUAL_LABOUR: employment or individual worker agreements with payroll/subordination cues.
REAL_ESTATE_AND_FACILITIES: leases, deeds, construction, fixed infrastructure.
IP_LICENSING_AND_TECH: SaaS, software licenses, platform access, IP licensing.
SUPPLY_AND_TRADE: sale, supply, distribution, logistics of physical goods.
SERVICES: consulting, professional services, statements of work, agency work.
LEGAL_SETTLEMENT_AND_RIGHTS: NDAs, settlements, releases, powers of attorney.
""".strip()
MONTH_MAP = {
    "january": 1,
    "janvier": 1,
    "february": 2,
    "fevrier": 2,
    "février": 2,
    "march": 3,
    "mars": 3,
    "april": 4,
    "avril": 4,
    "may": 5,
    "mai": 5,
    "june": 6,
    "juin": 6,
    "july": 7,
    "juillet": 7,
    "august": 8,
    "aout": 8,
    "août": 8,
    "september": 9,
    "septembre": 9,
    "october": 10,
    "octobre": 10,
    "november": 11,
    "novembre": 11,
    "december": 12,
    "decembre": 12,
    "décembre": 12,
}


def compact_whitespace(text: str) -> str:
    """Collapse repeated whitespace so quotes and matches are easier to compare."""

    return re.sub(r"\s+", " ", text).strip()


def split_clauses(contract_text: str) -> list[str]:
    """Split raw text into compact clause-like chunks."""

    return [
        compact_quote(chunk, limit=400)
        for chunk in re.split(r"\n{1,2}|(?<=[.;])\s+", contract_text)
        if chunk.strip()
    ]


def compact_quote(text: str, limit: int = 240) -> str:
    """Turn a raw text fragment into a short evidence quote for the output JSON."""

    quote = compact_whitespace(text)
    if len(quote) <= limit:
        return quote
    return quote[: limit - 3].rstrip() + "..."


def strip_accents(text: str) -> str:
    """Remove accents so French text can be matched with simpler regex patterns."""

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def fold_text(text: str) -> str:
    """Lowercase and de-accent text for forgiving bilingual keyword matching."""

    return strip_accents(text).lower()


def empty_evidence() -> dict[str, EvidenceItem]:
    """Build an empty evidence object so every field is always present in output."""

    return {
        field: EvidenceItem(quote=None, note=None, derived=False)
        for field in EVIDENCE_FIELDS
    }


def preprocess_contract(contract_text: str) -> PreparedContract:
    """Build a reusable normalized view of the contract for fast local search."""

    lines = contract_text.splitlines()
    folded_lines = [fold_text(line) for line in lines]
    clauses = split_clauses(contract_text)
    folded_clauses = [fold_text(clause) for clause in clauses]
    header_lines = lines[:40]
    header_text = "\n".join(header_lines)
    return PreparedContract(
        text=contract_text,
        lines=lines,
        folded_lines=folded_lines,
        clauses=clauses,
        folded_clauses=folded_clauses,
        header_text=header_text,
        header_lines=header_lines,
        folded_header=fold_text(header_text),
    )


def dedupe_warnings(warnings: list[str]) -> list[str]:
    """Keep warning messages readable by removing duplicates and blank entries."""

    seen: set[str] = set()
    result: list[str] = []
    for warning in warnings:
        normalized = warning.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def serialize_value(value: Any) -> Any:
    """Convert enums into plain JSON-friendly values."""

    if isinstance(value, Enum):
        return value.value
    return value


def is_locked_candidate(field: str, candidate: Optional[Candidate]) -> bool:
    """Return True when a local value is strong enough to bypass LLM review."""

    if candidate is None or candidate.value is None:
        return False
    threshold = LOCK_CONFIDENCE_BY_FIELD.get(field, 0.0)
    return candidate.confidence >= threshold


def should_emit_candidate(field: str, candidate: Optional[Candidate]) -> bool:
    """Return True when a candidate is strong enough to appear in final output."""

    if candidate is None or candidate.value is None:
        return False
    if candidate.source == "llm":
        return True
    return is_locked_candidate(field, candidate)


def store_candidate(
    candidates: dict[str, Candidate],
    field: str,
    candidate: Candidate,
    warnings: list[str],
) -> None:
    """Store a local match, resolving simple conflicts conservatively.

    If two local rules disagree about the same field, we drop the local answer
    and leave the field for the LLM or final null output. That is safer than
    picking one arbitrarily.
    """

    if candidate.value is None:
        return

    existing = candidates.get(field)
    if existing and existing.value is not None:
        if serialize_value(existing.value) != serialize_value(candidate.value):
            candidates[field] = Candidate()
            warnings.append(
                f"Conflicting local matches for {field}; leaving it for LLM resolution."
            )
            return
        if existing.confidence >= candidate.confidence:
            return

    candidates[field] = candidate


def safe_iso_date(year: int, month: int, day: int) -> Optional[str]:
    """Create an ISO date string, returning None for impossible dates."""

    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_date_string(raw: str) -> Optional[str]:
    """Normalize a date written in common English or French contract formats."""

    value = compact_whitespace(raw).strip(".,;:()[]")
    value = re.sub(r"(?i)(\d)(st|nd|rd|th)\b", r"\1", value)

    match = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", value)
    if match:
        return safe_iso_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", value)
    if match:
        first = int(match.group(1))
        second = int(match.group(2))
        year = int(match.group(3))
        if first > 12 and second <= 12:
            return safe_iso_date(year, second, first)
        if second > 12 and first <= 12:
            return safe_iso_date(year, first, second)
        return None

    match = re.fullmatch(r"(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", value)
    if match:
        day = int(match.group(1))
        month = MONTH_MAP.get(fold_text(match.group(2)))
        year = int(match.group(3))
        if month:
            return safe_iso_date(year, month, day)

    match = re.fullmatch(r"([A-Za-zÀ-ÿ]+)\s+(\d{1,2}),?\s+(\d{4})", value)
    if match:
        month = MONTH_MAP.get(fold_text(match.group(1)))
        day = int(match.group(2))
        year = int(match.group(3))
        if month:
            return safe_iso_date(year, month, day)

    return None


def parse_number(raw: str) -> Optional[float]:
    """Parse a money-like number while tolerating spaces, commas, and dots."""

    value = raw.strip().replace("\u00a0", " ")
    value = re.sub(r"[^\d,.\s]", "", value)
    value = re.sub(r"\s+", "", value)
    if not value:
        return None

    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif value.count(",") > 1:
        value = value.replace(",", "")
    elif value.count(".") > 1:
        value = value.replace(".", "")
    elif "," in value:
        whole, fraction = value.split(",", 1)
        value = whole + ("." + fraction if len(fraction) in (1, 2) else fraction)
    elif "." in value:
        whole, fraction = value.split(".", 1)
        value = whole + ("." + fraction if len(fraction) in (1, 2) else fraction)

    try:
        return float(value)
    except ValueError:
        return None


def detect_language(contract_text: str) -> Optional[Language]:
    """Estimate whether the document is mostly English, French, or mixed."""

    folded = fold_text(contract_text)
    fr_cues = [
        "le present",
        "contrat",
        "date d'effet",
        "parties",
        "societe",
        "durée",
        "duree",
        "renouvellement",
        "résiliation",
        "resiliation",
    ]
    en_cues = [
        "this agreement",
        "effective date",
        "whereas",
        "party",
        "parties",
        "term",
        "termination",
        "renewal",
    ]
    fr_score = sum(folded.count(cue) for cue in fr_cues)
    en_score = sum(folded.count(cue) for cue in en_cues)
    if fr_score == 0 and en_score == 0:
        return None
    if fr_score >= en_score * 1.5 and fr_score >= 2:
        return Language.FR
    if en_score >= fr_score * 1.5 and en_score >= 2:
        return Language.EN
    return Language.MIXED


def iter_clauses(contract_text: str) -> list[str]:
    """Compatibility helper that returns clause-like chunks from raw text."""

    return split_clauses(contract_text)


def extract_primary_contract_type(prepared: PreparedContract) -> Optional[Candidate]:
    """Guess the primary contract type from the document title or early lines.

    This is intentionally conservative. It only fires when the title contains
    strong type words like "services agreement" or "amendment".
    """

    for contract_type, keywords in PRIMARY_TYPE_RULES:
        for keyword in keywords:
            pattern = rf"(?<!\w){re.escape(fold_text(keyword))}(?!\w)"
            if re.search(pattern, prepared.folded_header):
                matched_line = next(
                    (
                        compact_quote(line)
                        for line in prepared.header_lines
                        if re.search(pattern, fold_text(line))
                    ),
                    compact_quote(prepared.header_text),
                )
                return Candidate(
                    value=contract_type,
                    quote=matched_line,
                    confidence=0.88,
                    source="regex",
                )
    return None


def extract_counterparty_contact_email(contract_text: str) -> Optional[Candidate]:
    """Extract an external contact email only when surrounding text supports it."""

    for clause in iter_clauses(contract_text):
        emails = sorted({match.group(0) for match in EMAIL_PATTERN.finditer(clause)})
        if len(emails) != 1:
            continue
        folded = fold_text(clause)
        if not any(cue in folded for cue in EXTERNAL_EMAIL_CUES):
            continue
        if any(cue in folded for cue in INTERNAL_EMAIL_CUES):
            continue
        return Candidate(
            value=emails[0],
            quote=compact_quote(clause),
            confidence=0.96,
            source="regex",
        )
    return None


def extract_labeled_date(contract_text: str, labels: list[str]) -> Optional[Candidate]:
    """Find a date that sits next to a clear label such as 'Effective Date'."""

    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"(?is)\b(?:{label_pattern})\b[^\n\r:\d]{{0,40}}[:\-]?\s*(?P<date>{DATE_VALUE_PATTERN})"
    )
    for match in pattern.finditer(contract_text):
        parsed = parse_date_string(match.group("date"))
        if parsed:
            return Candidate(
                value=parsed,
                quote=compact_quote(match.group(0)),
                confidence=0.95,
                source="regex",
            )
    return None


def extract_role_labeled_entity(
    contract_text: str,
    role_labels: list[str],
    *,
    field_name: str,
    confidence: float,
) -> Optional[Candidate]:
    """Extract a named party from lines labeled with an external-facing role."""

    label_pattern = "|".join(re.escape(label) for label in role_labels)
    pattern = re.compile(
        rf"(?im)^\s*(?:{label_pattern})\s*[:\-]\s*(?P<name>[^\n,;()]+(?:\([^)\n]+\))?)"
    )
    for match in pattern.finditer(contract_text):
        name = compact_whitespace(match.group("name")).strip(" -:")
        if len(name) < 3:
            continue
        return Candidate(
            value=name,
            quote=compact_quote(match.group(0)),
            note=f"Role-labeled {field_name} match.",
            confidence=confidence,
            source="regex",
        )
    return None


def extract_counterparty_name(prepared: PreparedContract) -> Optional[Candidate]:
    """Try to find the external contracting party using explicit role labels first."""

    candidate = extract_role_labeled_entity(
        prepared.text,
        [
            "customer",
            "client",
            "vendor",
            "supplier",
            "provider",
            "contractor",
            "consultant",
            "licensor",
            "licensee",
            "buyer",
            "seller",
            "prestataire",
            "fournisseur",
        ],
        field_name="counterparty name",
        confidence=0.96,
    )
    if candidate:
        return candidate

    pair_patterns = [
        re.compile(
            r"(?is)\bbetween\s+(?P<party1>[^,\n;()]+?)\s+and\s+(?P<party2>[^,\n;()]+?)(?:\s*\(|,|\n|$)"
        ),
        re.compile(
            r"(?is)\bentre\s+(?P<party1>[^,\n;()]+?)\s+et\s+(?P<party2>[^,\n;()]+?)(?:\s*\(|,|\n|$)"
        ),
    ]
    for pattern in pair_patterns:
        match = pattern.search(prepared.text[:4000])
        if not match:
            continue
        party2 = compact_whitespace(match.group("party2")).strip(" -:")
        if len(party2) < 3:
            continue
        return Candidate(
            value=party2,
            quote=compact_quote(match.group(0)),
            note="Fallback assumption: second named party in a two-party definition.",
            confidence=0.7,
            source="regex",
        )
    return None


def extract_counterparty_signatory(prepared: PreparedContract) -> Optional[Candidate]:
    """Search signature blocks for an external signatory name."""

    role_blocks = [
        re.compile(
            r"(?is)(?:for and on behalf of|on behalf of|pour le compte de)\s+"
            r"(?:customer|client|vendor|supplier|provider|contractor|consultant|prestataire|fournisseur)[^\n]*"
            r"(?:\n.{0,120}){0,4}?\b(?:name|nom)\s*[:\-]\s*(?P<person>[A-Z][^\n,:;]{2,80})"
        ),
        re.compile(
            r"(?is)^\s*(?:customer|client|vendor|supplier|provider|contractor|consultant|prestataire|fournisseur)\s*[:\-].*?"
            r"(?:\n.{0,120}){0,4}?\b(?:by|name|nom)\s*[:\-]\s*(?P<person>[A-Z][^\n,:;]{2,80})",
            re.MULTILINE,
        ),
    ]
    for pattern in role_blocks:
        match = pattern.search(prepared.text)
        if not match:
            continue
        person = compact_whitespace(match.group("person")).strip(" -:")
        if len(person.split()) < 2:
            continue
        return Candidate(
            value=person,
            quote=compact_quote(match.group(0)),
            confidence=0.95,
            source="regex",
        )
    return None


def extract_clause_based_candidates(
    prepared: PreparedContract, warnings: list[str]
) -> dict[str, Candidate]:
    """Extract several usually-present fields in one clause scan.

    This function is part of the speed optimization work: instead of looping
    over all clauses once per field, it inspects each clause once and tries to
    fill multiple related fields during that pass.
    """

    candidates: dict[str, Candidate] = {}
    periodicity_matches: list[tuple[PaymentPeriodicity, str]] = []
    periodicity_patterns: list[tuple[PaymentPeriodicity, list[str]]] = [
        (
            PaymentPeriodicity.MONTHLY,
            [
                r"\bbilled monthly\b",
                r"\binvoiced monthly\b",
                r"\binvoices? (?:are )?issued monthly\b",
                r"\bpaid monthly\b",
                r"\bpayable monthly\b",
                r"\bmonthly (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bper month\b",
                r"\bmensuel\b",
                r"\bmensuellement\b",
                r"\bpar mois\b",
            ],
        ),
        (
            PaymentPeriodicity.QUARTERLY,
            [
                r"\bbilled quarterly\b",
                r"\binvoiced quarterly\b",
                r"\binvoices? (?:are )?issued quarterly\b",
                r"\bpaid quarterly\b",
                r"\bpayable quarterly\b",
                r"\bquarterly (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bper quarter\b",
                r"\btrimestriel\b",
                r"\btrimestriellement\b",
            ],
        ),
        (
            PaymentPeriodicity.SEMI_ANNUAL,
            [
                r"\bbilled semi[- ]annually\b",
                r"\binvoiced semi[- ]annually\b",
                r"\binvoices? (?:are )?issued semi[- ]annually\b",
                r"\bpaid semi[- ]annually\b",
                r"\bpayable semi[- ]annually\b",
                r"\bsemi[- ]annual (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bsemi[- ]annual\b",
                r"\bsemiannual\b",
                r"\bsemi-annuel\b",
            ],
        ),
        (
            PaymentPeriodicity.ANNUAL,
            [
                r"\bbilled annually\b",
                r"\binvoiced annually\b",
                r"\binvoices? (?:are )?issued annually\b",
                r"\bpaid annually\b",
                r"\bpayable annually\b",
                r"\bannual (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bper year\b",
                r"\bannuel\b",
                r"\bannuellement\b",
            ],
        ),
        (PaymentPeriodicity.MILESTONE_BASED, [r"\bmilestone\b", r"\bjalon\b"]),
        (PaymentPeriodicity.USAGE_BASED, [r"\busage[- ]based\b", r"\bper api call\b", r"\bper use\b", r"\bselon l'utilisation\b"]),
        (PaymentPeriodicity.ON_DELIVERY, [r"\bon delivery\b", r"\bupon delivery\b", r"\ba la livraison\b", r"\bà la livraison\b"]),
        (PaymentPeriodicity.ONE_OFF, [r"\bone[- ]off\b", r"\bone time\b", r"\bsingle payment\b", r"\blump sum\b", r"\bforfait unique\b"]),
    ]
    payment_terms_patterns = [
        r"\bnet\s+(\d{1,3})\b",
        r"\bwithin\s+\w+\s*\((\d{1,3})\)\s+days\b",
        r"\bwithin\s+(\d{1,3})\s+days\b",
        r"\bpayable\s+within\s+(\d{1,3})\s+days\b",
        r"\bdans un delai de\s+(\d{1,3})\s+jours\b",
        r"\bpaiement sous\s+(\d{1,3})\s+jours\b",
    ]
    renewal_negative_patterns = [
        r"\bno automatic renewal\b",
        r"\bwithout automatic renewal\b",
        r"\bshall not renew automatically\b",
        r"\bwill not renew automatically\b",
        r"\bsans reconduction tacite\b",
        r"\bsans renouvellement automatique\b",
        r"\bne sera pas renouvele automatiquement\b",
    ]
    renewal_positive_patterns = [
        r"\bautomatic renewal\b",
        r"\bautomatically renew",
        r"\brenew automatically\b",
        r"\bevergreen\b",
        r"\btacite reconduction\b",
        r"\brenouvellement automatique\b",
    ]
    notice_patterns = [
        r"\b(\d{1,3})\s+days?\b",
        r"\b(\d{1,3})\s+calendar\s+days?\b",
        r"\bpreavis\s+de\s+(\d{1,3})\s+jours\b",
        r"\bdelai\s+de\s+preavis\s+de\s+(\d{1,3})\s+jours\b",
    ]

    for clause, folded in zip(prepared.clauses, prepared.folded_clauses):
        if "auto_renewal" not in candidates:
            if any(re.search(pattern, folded) for pattern in renewal_negative_patterns):
                candidates["auto_renewal"] = Candidate(
                    value=False, quote=clause, confidence=0.95, source="regex"
                )
            elif any(re.search(pattern, folded) for pattern in renewal_positive_patterns):
                candidates["auto_renewal"] = Candidate(
                    value=True, quote=clause, confidence=0.95, source="regex"
                )

        if "renewal_notice_days" not in candidates:
            if any(keyword in folded for keyword in ("renew", "renewal", "reconduction", "renouvellement")) and "month" not in folded and "mois" not in folded:
                for pattern in notice_patterns:
                    match = re.search(pattern, folded)
                    if match:
                        candidates["renewal_notice_days"] = Candidate(
                            value=int(match.group(1)),
                            quote=clause,
                            confidence=0.9,
                            source="regex",
                        )
                        break

        if "termination_notice_days" not in candidates:
            if any(keyword in folded for keyword in ("terminate", "termination", "resiliation", "résiliation", "terminate this agreement")) and "month" not in folded and "mois" not in folded:
                for pattern in notice_patterns:
                    match = re.search(pattern, folded)
                    if match:
                        candidates["termination_notice_days"] = Candidate(
                            value=int(match.group(1)),
                            quote=clause,
                            confidence=0.9,
                            source="regex",
                        )
                        break

        if "contract_term" not in candidates and any(
            token in folded for token in ("term", "duration", "duree", "durée")
        ):
            if any(
                token in folded
                for token in (
                    "indefinite",
                    "until terminated",
                    "duree indeterminee",
                    "durée indéterminée",
                )
            ):
                candidates["contract_term"] = Candidate(
                    value="indefinite", quote=clause, confidence=0.92, source="regex"
                )
            else:
                match = re.search(
                    r"\b(?:initial\s+term|term|duration)\b.{0,60}?\b(\d{1,3})\s+(months?|years?)\b.{0,40}?\bfrom\s+(effective date|start date|commencement date)\b",
                    folded,
                )
                if match:
                    unit = "months" if match.group(2).startswith("month") else "years"
                    base = {
                        "effective date": "Effective Date",
                        "start date": "Start Date",
                        "commencement date": "Start Date",
                    }[match.group(3)]
                    candidates["contract_term"] = Candidate(
                        value=f"{int(match.group(1))} {unit} from {base}",
                        quote=clause,
                        confidence=0.9,
                        source="regex",
                    )
                else:
                    match = re.search(
                        r"\b(?:duree|durée)\b.{0,40}?\b(\d{1,3})\s+(mois|ans?|annees?|années?)\b.{0,40}?\ba compter de\b.{0,20}?\b(date d'effet|date de debut|date de début|prise d'effet)\b",
                        folded,
                    )
                    if match:
                        unit = "months" if match.group(2) == "mois" else "years"
                        base = "Effective Date" if "effet" in match.group(3) else "Start Date"
                        candidates["contract_term"] = Candidate(
                            value=f"{int(match.group(1))} {unit} from {base}",
                            quote=clause,
                            confidence=0.9,
                            source="regex",
                        )
                    else:
                        match = re.search(
                            rf"\buntil\b\s+(?P<date>{DATE_VALUE_PATTERN})",
                            clause,
                            re.IGNORECASE,
                        )
                        if match:
                            parsed = parse_date_string(match.group("date"))
                            if parsed:
                                candidates["contract_term"] = Candidate(
                                    value=f"until {parsed}",
                                    quote=clause,
                                    confidence=0.88,
                                    source="regex",
                                )

        if any(cue in folded for cue in PAYMENT_CLAUSE_CUES):
            if "payment_terms_days" not in candidates:
                for pattern in payment_terms_patterns:
                    match = re.search(pattern, folded)
                    if match:
                        candidates["payment_terms_days"] = Candidate(
                            value=int(match.group(1)),
                            quote=clause,
                            confidence=0.95,
                            source="regex",
                        )
                        break

            if not (
                any(cue in folded for cue in INDEXATION_CUES)
                and any(
                    cue in folded
                    for cue in ("increase", "adjust", "uplift", "revision", "révision")
                )
            ):
                for periodicity, patterns in periodicity_patterns:
                    if any(re.search(pattern, folded) for pattern in patterns):
                        periodicity_matches.append((periodicity, clause))

    if periodicity_matches and "payment_periodicity" not in candidates:
        unique_values = {periodicity for periodicity, _ in periodicity_matches}
        if len(unique_values) == 1:
            periodicity, quote = periodicity_matches[0]
            candidates["payment_periodicity"] = Candidate(
                value=periodicity,
                quote=quote,
                confidence=0.92,
                source="regex",
            )
        else:
            warnings.append(
                "Multiple payment cadences detected locally; returning OTHER for payment_periodicity."
            )
            candidates["payment_periodicity"] = Candidate(
                value=PaymentPeriodicity.OTHER,
                quote=periodicity_matches[0][1],
                note="Multiple explicit billing cadences detected.",
                confidence=0.7,
                source="regex",
            )

    return candidates


def extract_date_candidates(contract_text: str) -> dict[str, Candidate]:
    """Run the labeled date extractor for the main contract date fields."""

    return {
        "signature_date": extract_labeled_date(
            contract_text,
            ["signature date", "date of signature", "executed on", "signed on", "execution date"],
        )
        or Candidate(),
        "effective_date": extract_labeled_date(
            contract_text,
            ["effective date", "date d'effet", "prise d'effet", "entry into force", "entrée en vigueur"],
        )
        or Candidate(),
        "start_date": extract_labeled_date(
            contract_text,
            ["start date", "commencement date", "service commencement", "date de début", "date de debut"],
        )
        or Candidate(),
        "end_date": extract_labeled_date(
            contract_text,
            ["end date", "expiry date", "expiration date", "date de fin", "échéance", "echeance"],
        )
        or Candidate(),
    }


def extract_auto_renewal(contract_text: str) -> Optional[Candidate]:
    """Detect explicit auto-renewal or explicit no-auto-renewal language."""

    negative_patterns = [
        r"\bno automatic renewal\b",
        r"\bwithout automatic renewal\b",
        r"\bshall not renew automatically\b",
        r"\bwill not renew automatically\b",
        r"\bsans reconduction tacite\b",
        r"\bsans renouvellement automatique\b",
        r"\bne sera pas renouvele automatiquement\b",
    ]
    positive_patterns = [
        r"\bautomatic renewal\b",
        r"\bautomatically renew",
        r"\brenew automatically\b",
        r"\bevergreen\b",
        r"\btacite reconduction\b",
        r"\brenouvellement automatique\b",
    ]
    for clause in iter_clauses(contract_text):
        folded = fold_text(clause)
        if any(re.search(pattern, folded) for pattern in negative_patterns):
            return Candidate(value=False, quote=clause, confidence=0.95, source="regex")
        if any(re.search(pattern, folded) for pattern in positive_patterns):
            return Candidate(value=True, quote=clause, confidence=0.95, source="regex")
    return None


def extract_notice_days(contract_text: str, renewal: bool) -> Optional[Candidate]:
    """Extract notice periods in days for renewal or termination clauses."""

    keywords = (
        ("renew", "renewal", "reconduction", "renouvellement")
        if renewal
        else ("terminate", "termination", "resiliation", "résiliation", "terminate this agreement")
    )
    patterns = [
        r"\b(\d{1,3})\s+days?\b",
        r"\b(\d{1,3})\s+calendar\s+days?\b",
        r"\bpreavis\s+de\s+(\d{1,3})\s+jours\b",
        r"\bdelai\s+de\s+preavis\s+de\s+(\d{1,3})\s+jours\b",
    ]
    for clause in iter_clauses(contract_text):
        folded = fold_text(clause)
        if not any(keyword in folded for keyword in keywords):
            continue
        if "month" in folded or "mois" in folded:
            continue
        for pattern in patterns:
            match = re.search(pattern, folded)
            if match:
                return Candidate(
                    value=int(match.group(1)),
                    quote=clause,
                    confidence=0.9,
                    source="regex",
                )
    return None


def extract_contract_term(contract_text: str) -> Optional[Candidate]:
    """Normalize explicit term language into a short human-readable summary."""

    for clause in iter_clauses(contract_text):
        folded = fold_text(clause)
        if not any(token in folded for token in ("term", "duration", "duree", "durée")):
            continue
        if any(token in folded for token in ("indefinite", "until terminated", "duree indeterminee", "durée indéterminée")):
            return Candidate(
                value="indefinite",
                quote=clause,
                confidence=0.92,
                source="regex",
            )

        match = re.search(
            r"\b(?:initial\s+term|term|duration)\b.{0,60}?\b(\d{1,3})\s+(months?|years?)\b.{0,40}?\bfrom\s+(effective date|start date|commencement date)\b",
            folded,
        )
        if match:
            unit = "months" if match.group(2).startswith("month") else "years"
            base = {
                "effective date": "Effective Date",
                "start date": "Start Date",
                "commencement date": "Start Date",
            }[match.group(3)]
            return Candidate(
                value=f"{int(match.group(1))} {unit} from {base}",
                quote=clause,
                confidence=0.9,
                source="regex",
            )

        match = re.search(
            r"\b(?:duree|durée)\b.{0,40}?\b(\d{1,3})\s+(mois|ans?|annees?|années?)\b.{0,40}?\ba compter de\b.{0,20}?\b(date d'effet|date de debut|date de début|prise d'effet)\b",
            folded,
        )
        if match:
            unit = "months" if match.group(2) == "mois" else "years"
            base = "Effective Date" if "effet" in match.group(3) else "Start Date"
            return Candidate(
                value=f"{int(match.group(1))} {unit} from {base}",
                quote=clause,
                confidence=0.9,
                source="regex",
            )

        match = re.search(rf"\buntil\b\s+(?P<date>{DATE_VALUE_PATTERN})", clause, re.IGNORECASE)
        if match:
            parsed = parse_date_string(match.group("date"))
            if parsed:
                return Candidate(
                    value=f"until {parsed}",
                    quote=clause,
                    confidence=0.88,
                    source="regex",
                )
    return None


def extract_term_year_multiplier(contract_text: str) -> Optional[tuple[float, str]]:
    """Return an exact year multiplier only when the committed term is explicit."""

    for clause in iter_clauses(contract_text):
        folded = fold_text(clause)
        if not any(token in folded for token in ("term", "duration", "duree", "durée")):
            continue

        year_match = re.search(
            r"\b(\d{1,3})\s+(years?|ans?|annees?|années?)\b",
            folded,
        )
        if year_match:
            return float(int(year_match.group(1))), clause

        month_match = re.search(
            r"\b(\d{1,3})\s+(months?|mois)\b",
            folded,
        )
        if month_match:
            months = int(month_match.group(1))
            if months % 12 == 0:
                return months / 12, clause
    return None


def extract_total_value(contract_text: str) -> dict[str, Candidate]:
    """Extract an explicitly labeled total contract value and its currency.

    This function does not try to be clever with incomplete pricing tables.
    It only accepts totals that are written directly in the contract.
    """

    results: dict[str, Candidate] = {}
    labels = [
        "total fees",
        "total fee",
        "total amount",
        "contract value",
        "total contract value",
        "total price",
        "montant total",
        "prix total",
        "valeur totale du contrat",
        "minimum commitment",
        "minimum annual commitment",
    ]
    currency_pattern = r"USD|EUR|GBP|CHF|CAD|AUD|JPY|US\$|C\$|€|£"
    amount_pattern = r"\d{1,3}(?:[ \u00A0.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
    compiled_patterns = [
        re.compile(
            rf"(?is)\b(?:{'|'.join(re.escape(label) for label in labels)})\b[^\n]{{0,30}}?(?P<currency>{currency_pattern})\s*(?P<amount>{amount_pattern})"
        ),
        re.compile(
            rf"(?is)\b(?:{'|'.join(re.escape(label) for label in labels)})\b[^\n]{{0,30}}?(?P<amount>{amount_pattern})\s*(?P<currency>{currency_pattern})"
        ),
    ]
    for pattern in compiled_patterns:
        match = pattern.search(contract_text)
        if not match:
            continue
        amount = parse_number(match.group("amount"))
        if amount is None:
            continue
        quote = compact_quote(match.group(0))
        currency = normalize_currency_token(match.group("currency"))
        label_text = fold_text(match.group(0))
        derived = False
        note = None
        if "annual commitment" in label_text:
            term_multiplier = extract_term_year_multiplier(contract_text)
            if term_multiplier is None:
                continue
            years, term_quote = term_multiplier
            amount = amount * years
            derived = years != 1
            note = (
                f"Annual commitment multiplied by explicit committed term ({term_quote})"
                if derived
                else "Explicit annual commitment for a 12-month committed term."
            )
            quote = compact_quote(f"{match.group(0)} | {term_quote}")
        results["total_contract_value"] = Candidate(
            value=amount,
            quote=quote,
            note=note,
            derived=derived,
            confidence=0.96,
            source="regex",
        )
        if currency:
            results["currency"] = Candidate(
                value=currency,
                quote=quote,
                confidence=0.95,
                source="regex",
            )
        return results
    return results


def normalize_currency_token(token: Optional[str]) -> Optional[str]:
    """Map symbols and common abbreviations to a single ISO-like currency code."""

    if not token:
        return None
    mapping = {
        "EUR": "EUR",
        "USD": "USD",
        "GBP": "GBP",
        "CHF": "CHF",
        "CAD": "CAD",
        "AUD": "AUD",
        "JPY": "JPY",
        "US$": "USD",
        "C$": "CAD",
        "€": "EUR",
        "£": "GBP",
    }
    return mapping.get(token.upper() if token not in {"€", "£"} else token)


def extract_currency(contract_text: str) -> Optional[Candidate]:
    """Return a currency only when the document appears to use one clear currency."""

    tokens = re.findall(r"\b(?:USD|EUR|GBP|CHF|CAD|AUD|JPY)\b|US\$|C\$|€|£", contract_text)
    normalized = {normalize_currency_token(token) for token in tokens if normalize_currency_token(token)}
    if len(normalized) == 1:
        currency = normalized.pop()
        return Candidate(value=currency, quote=currency, confidence=0.75, source="regex")
    return None


def extract_payment_periodicity(contract_text: str, warnings: list[str]) -> Optional[Candidate]:
    """Infer the billing cadence from payment clauses.

    If the text clearly mentions more than one cadence, we fall back to OTHER
    and emit a warning instead of pretending the answer is simple.
    """

    periodicity_patterns: list[tuple[PaymentPeriodicity, list[str]]] = [
        (
            PaymentPeriodicity.MONTHLY,
            [
                r"\bbilled monthly\b",
                r"\binvoiced monthly\b",
                r"\binvoices? (?:are )?issued monthly\b",
                r"\bpaid monthly\b",
                r"\bpayable monthly\b",
                r"\bmonthly (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bper month\b",
                r"\bmensuel\b",
                r"\bmensuellement\b",
                r"\bpar mois\b",
            ],
        ),
        (
            PaymentPeriodicity.QUARTERLY,
            [
                r"\bbilled quarterly\b",
                r"\binvoiced quarterly\b",
                r"\binvoices? (?:are )?issued quarterly\b",
                r"\bpaid quarterly\b",
                r"\bpayable quarterly\b",
                r"\bquarterly (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bper quarter\b",
                r"\btrimestriel\b",
                r"\btrimestriellement\b",
            ],
        ),
        (
            PaymentPeriodicity.SEMI_ANNUAL,
            [
                r"\bbilled semi[- ]annually\b",
                r"\binvoiced semi[- ]annually\b",
                r"\binvoices? (?:are )?issued semi[- ]annually\b",
                r"\bpaid semi[- ]annually\b",
                r"\bpayable semi[- ]annually\b",
                r"\bsemi[- ]annual (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bsemi[- ]annual\b",
                r"\bsemiannual\b",
                r"\bsemi-annuel\b",
            ],
        ),
        (
            PaymentPeriodicity.ANNUAL,
            [
                r"\bbilled annually\b",
                r"\binvoiced annually\b",
                r"\binvoices? (?:are )?issued annually\b",
                r"\bpaid annually\b",
                r"\bpayable annually\b",
                r"\bannual (?:fee|fees|payment|payments|invoice|billing)\b",
                r"\bper year\b",
                r"\bannuel\b",
                r"\bannuellement\b",
            ],
        ),
        (PaymentPeriodicity.MILESTONE_BASED, [r"\bmilestone\b", r"\bjalon\b"]),
        (PaymentPeriodicity.USAGE_BASED, [r"\busage[- ]based\b", r"\bper api call\b", r"\bper use\b", r"\bselon l'utilisation\b"]),
        (PaymentPeriodicity.ON_DELIVERY, [r"\bon delivery\b", r"\bupon delivery\b", r"\ba la livraison\b", r"\bà la livraison\b"]),
        (PaymentPeriodicity.ONE_OFF, [r"\bone[- ]off\b", r"\bone time\b", r"\bsingle payment\b", r"\blump sum\b", r"\bforfait unique\b"]),
    ]
    matches: list[tuple[PaymentPeriodicity, str]] = []
    for clause in iter_clauses(contract_text):
        folded = fold_text(clause)
        if not any(cue in folded for cue in PAYMENT_CLAUSE_CUES):
            continue
        if any(cue in folded for cue in INDEXATION_CUES) and any(
            cue in folded for cue in ("increase", "adjust", "uplift", "revision", "révision")
        ):
            continue
        for periodicity, patterns in periodicity_patterns:
            if any(re.search(pattern, folded) for pattern in patterns):
                matches.append((periodicity, clause))
    if not matches:
        return None

    unique_values = {periodicity for periodicity, _ in matches}
    if len(unique_values) == 1:
        periodicity, quote = matches[0]
        return Candidate(value=periodicity, quote=quote, confidence=0.92, source="regex")

    warnings.append("Multiple payment cadences detected locally; returning OTHER for payment_periodicity.")
    return Candidate(
        value=PaymentPeriodicity.OTHER,
        quote=matches[0][1],
        note="Multiple explicit billing cadences detected.",
        confidence=0.7,
        source="regex",
    )


def extract_payment_terms_days(contract_text: str) -> Optional[Candidate]:
    """Extract invoice payment terms such as Net 30 or payable within 45 days."""

    patterns = [
        r"\bnet\s+(\d{1,3})\b",
        r"\bwithin\s+\w+\s*\((\d{1,3})\)\s+days\b",
        r"\bwithin\s+(\d{1,3})\s+days\b",
        r"\bpayable\s+within\s+(\d{1,3})\s+days\b",
        r"\bdans un delai de\s+(\d{1,3})\s+jours\b",
        r"\bpaiement sous\s+(\d{1,3})\s+jours\b",
    ]
    for clause in iter_clauses(contract_text):
        folded = fold_text(clause)
        if not any(cue in folded for cue in PAYMENT_CLAUSE_CUES):
            continue
        for pattern in patterns:
            match = re.search(pattern, folded)
            if match:
                return Candidate(
                    value=int(match.group(1)),
                    quote=clause,
                    confidence=0.95,
                    source="regex",
                )
    return None


def extract_indexation(contract_text: str) -> Optional[Candidate]:
    """Detect explicit price adjustment mechanisms like CPI or fixed annual uplift."""

    for clause in iter_clauses(contract_text):
        folded = fold_text(clause)
        if not any(
            cue in folded
            for cue in ("cpi", "ipc", "ilc", "ilat", "icc", "syntec", "inflation", "indexation", "revision", "révision", "increase by")
        ):
            continue
        if "syntec" in folded:
            return Candidate(value="Syntec index", quote=clause, confidence=0.95, source="regex")
        if "ilc" in folded:
            return Candidate(value="ILC index", quote=clause, confidence=0.95, source="regex")
        if "ilat" in folded:
            return Candidate(value="ILAT index", quote=clause, confidence=0.95, source="regex")
        if "icc" in folded:
            return Candidate(value="ICC index", quote=clause, confidence=0.95, source="regex")
        if "cpi" in folded or "consumer price index" in folded or "ipc" in folded:
            return Candidate(value="Annual CPI adjustment", quote=clause, confidence=0.92, source="regex")
        if "inflation" in folded:
            return Candidate(
                value="Annual inflation-linked adjustment",
                quote=clause,
                confidence=0.9,
                source="regex",
            )
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", clause)
        if match and any(token in folded for token in ("annual", "anniversary", "chaque annee", "chaque année", "each year", "every year")):
            pct = match.group(1).replace(",", ".")
            if pct.endswith(".0"):
                pct = pct[:-2]
            return Candidate(
                value=f"Fixed {pct} percent annual increase",
                quote=clause,
                confidence=0.9,
                source="regex",
            )
    return None


def extract_regex_candidates(contract_text: str) -> tuple[dict[str, Candidate], list[str]]:
    """Run the local-first extraction pass.

    This is the main "cheap checks first" stage. It preprocesses the contract
    once, then applies deterministic rules only to the fields that are usually
    explicit in the text. It returns:
    - the field candidates found locally
    - any warnings produced by conflicts or ambiguous local situations
    """

    prepared = preprocess_contract(contract_text)
    candidates: dict[str, Candidate] = {}
    warnings: list[str] = []

    primary_type = extract_primary_contract_type(prepared)
    if primary_type:
        store_candidate(candidates, "primary_contract_type", primary_type, warnings)

    counterparty_name = extract_counterparty_name(prepared)
    if counterparty_name:
        store_candidate(candidates, "counterparty_name", counterparty_name, warnings)

    counterparty_signatory = extract_counterparty_signatory(prepared)
    if counterparty_signatory:
        store_candidate(
            candidates, "counterparty_signatory", counterparty_signatory, warnings
        )

    for field, candidate in extract_date_candidates(contract_text).items():
        store_candidate(candidates, field, candidate, warnings)

    for field, candidate in extract_clause_based_candidates(prepared, warnings).items():
        store_candidate(candidates, field, candidate, warnings)

    currency = extract_currency(prepared.text)
    if currency:
        store_candidate(candidates, "currency", currency, warnings)

    return candidates, dedupe_warnings(warnings)


def build_prompt(
    contract_snippets: str,
    requested_fields: list[str],
    locked_values: dict[str, Any],
) -> str:
    """Build a smaller LLM prompt focused only on unresolved fields.

    The prompt tells the model:
    - which fields are still missing
    - which values were already locked in locally
    - which text snippets are relevant for the requested fields
    - only the fields the LLM is actually allowed to return
    """

    field_rules = "\n".join(
        f"- {field}: {LLM_FIELD_RULES[field]}" for field in requested_fields
    )
    prompt_parts = [
        "You are a bilingual legal contract extraction engine for English and French contracts.",
        "Extract only the requested fields from the provided snippets.",
        "Be conservative: never guess, never use outside knowledge, and return null if the value is absent, ambiguous, conflicting, or weakly implied.",
        "Return JSON only and match the schema exactly.",
        "For every non-null requested field with evidence, provide a short verbatim quote from the snippets.",
        "For every field that is not in requested_fields, return null and empty evidence.",
        f"requested_fields: {json.dumps(requested_fields)}",
        f"locked_values: {json.dumps(locked_values, ensure_ascii=False)}",
    ]
    if "primary_contract_type" in requested_fields:
        prompt_parts.append("Contract type taxonomy:\n" + PRIMARY_TYPE_TAXONOMY)
    prompt_parts.append("Field rules:\n" + field_rules)
    prompt_parts.append("<contract_snippets>\n" + contract_snippets + "\n</contract_snippets>")
    return "\n\n".join(prompt_parts)


def normalize_content(content: Any) -> str:
    """Flatten the model's response content into plain text before JSON parsing."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    """Recover a JSON object even if the model wraps it in markdown fences."""

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    starts = [index for index, ch in enumerate(text) if ch == "{"]
    if not starts:
        raise ValueError("No JSON object found in model output")

    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("Could not recover a valid JSON object")


def call_openrouter(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Send one request to OpenRouter and raise a clear error on HTTP failure."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=300)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"{resp.status_code}: {resp.text}", response=resp)
    return resp.json()


@lru_cache(maxsize=128)
def build_llm_response_model(requested_fields_key: tuple[str, ...]) -> type[BaseModel]:
    """Create and cache a minimal response model for the requested LLM fields."""

    evidence_fields = {
        field: (EvidenceItem, ...)
        for field in requested_fields_key
        if field in EVIDENCE_FIELDS
    }
    evidence_model = create_model(
        f"EvidenceSubset_{abs(hash(requested_fields_key))}",
        __config__=ConfigDict(extra="forbid"),
        **evidence_fields,
    )
    llm_fields: dict[str, tuple[Any, Any]] = {}
    for field in requested_fields_key:
        llm_fields[field] = (ContractExtraction.model_fields[field].annotation, None)
    llm_fields["evidence"] = (evidence_model, ...)
    llm_fields["warnings"] = (list[str], Field(default_factory=list))
    return create_model(
        f"ContractExtractionSubset_{abs(hash(requested_fields_key))}",
        __config__=ConfigDict(extra="forbid"),
        **llm_fields,
    )


def parse_response(raw: dict[str, Any], response_model: type[BaseModel]) -> BaseModel:
    """Parse and validate the model response against the expected schema."""

    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected response shape: {e}") from e

    text = normalize_content(content)
    data = extract_json_object(text)
    return response_model.model_validate(data)


def candidate_from_llm(result: BaseModel, field: str) -> Candidate:
    """Convert one field from the validated LLM subset into an internal candidate."""

    if not hasattr(result, field):
        return Candidate()
    value = getattr(result, field)
    if value is None:
        return Candidate()
    if field in EVIDENCE_FIELDS:
        evidence_item = getattr(result.evidence, field)
        return Candidate(
            value=value,
            quote=evidence_item.quote,
            note=evidence_item.note,
            derived=evidence_item.derived,
            confidence=0.6,
            source="llm",
        )
    return Candidate(value=value, confidence=0.6, source="llm")


def build_result_from_candidates(
    candidates: dict[str, Candidate], warnings: list[str]
) -> ContractExtraction:
    """Turn merged candidates into the final validated output object."""

    data: dict[str, Any] = {field: None for field in ALL_FIELDS}
    for field in ALL_FIELDS:
        candidate = candidates.get(field)
        if should_emit_candidate(field, candidate):
            data[field] = serialize_value(candidate.value)

    evidence = empty_evidence()
    for field in EVIDENCE_FIELDS:
        candidate = candidates.get(field)
        if should_emit_candidate(field, candidate):
            evidence[field] = EvidenceItem(
                quote=candidate.quote,
                note=candidate.note,
                derived=candidate.derived,
            )

    data["evidence"] = evidence
    data["warnings"] = dedupe_warnings(warnings)
    return ContractExtraction.model_validate(data)


def get_missing_fields(candidates: dict[str, Candidate]) -> list[str]:
    """List the fields that still need help after the local extraction pass."""

    missing: list[str] = []
    for field in ROUTED_FIELDS:
        candidate = candidates.get(field)
        if not is_locked_candidate(field, candidate):
            missing.append(field)
    return missing


def get_locked_values(candidates: dict[str, Candidate]) -> dict[str, Any]:
    """Collect already-resolved values so the LLM knows not to change them."""

    return {
        field: serialize_value(candidate.value)
        for field, candidate in candidates.items()
        if is_locked_candidate(field, candidate)
    }


def collect_relevant_snippets(
    contract_text: str, requested_fields: list[str], max_chars: int = 12000
) -> str:
    """Pick the most relevant lines for the unresolved fields.

    Instead of sending the whole contract to the LLM, this function gathers
    only lines that are likely to help with the still-missing fields. That
    keeps the prompt smaller and more focused.
    """

    prepared = preprocess_contract(contract_text)
    lines = prepared.lines
    if not lines:
        return contract_text[:max_chars]

    selected: set[int] = set()
    high_context_fields = {
        "language",
        "primary_contract_type",
        "internal_entity",
        "counterparty_name",
        "counterparty_signatory",
    }
    if any(field in high_context_fields for field in requested_fields):
        selected.update(range(min(60, len(lines))))
        selected.update(range(max(0, len(lines) - 60), len(lines)))

    for field in requested_fields:
        keywords = [fold_text(keyword) for keyword in FIELD_KEYWORDS.get(field, [])]
        if not keywords:
            continue
        for index, line in enumerate(lines):
            folded_line = prepared.folded_lines[index]
            if any(keyword in folded_line for keyword in keywords):
                selected.update(range(max(0, index - 1), min(len(lines), index + 2)))

    if not selected:
        return contract_text[:max_chars]

    ordered = sorted(selected)
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for index in ordered[1:]:
        if index == end + 1:
            end = index
            continue
        ranges.append((start, end))
        start = index
        end = index
    ranges.append((start, end))

    parts: list[str] = []
    for start, end in ranges:
        block = "\n".join(
            f"{line_no + 1}: {lines[line_no]}"
            for line_no in range(start, end + 1)
            if lines[line_no].strip()
        )
        if block:
            parts.append(block)

    snippets = "\n\n".join(parts)
    if len(snippets) <= max_chars:
        return snippets
    marker = "\n\n...\n\n"
    head_chars = max((max_chars - len(marker)) // 2, 0)
    tail_chars = max(max_chars - len(marker) - head_chars, 0)
    return snippets[:head_chars].rstrip() + marker + snippets[-tail_chars:].lstrip()


def build_payload(
    prompt: str, structured: bool, response_model: type[BaseModel]
) -> dict[str, Any]:
    """Create the OpenRouter request payload for the expected response model."""

    payload: dict[str, Any] = {
        "model": MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    if structured:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "contract_extraction",
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    return payload


def call_llm_for_missing_fields(
    contract_text: str,
    requested_fields: list[str],
    locked_values: dict[str, Any],
    api_key: str,
) -> BaseModel:
    """Run the fallback LLM call for unresolved fields only.

    It first tries structured output using a small field-specific schema. If
    the provider does not support structured mode or returns malformed JSON,
    it retries once with plain JSON-object mode.
    """

    response_model = build_llm_response_model(tuple(requested_fields))
    prompt = build_prompt(
        collect_relevant_snippets(contract_text, requested_fields),
        requested_fields,
        locked_values,
    )
    try:
        raw = call_openrouter(build_payload(prompt, structured=True, response_model=response_model), api_key)
        return parse_response(raw, response_model)
    except requests.HTTPError as e:
        msg = str(e).lower()
        if (
            "response_format" not in msg
            and "structured" not in msg
            and "json_schema" not in msg
        ):
            raise
    except (ValueError, ValidationError, json.JSONDecodeError):
        pass

    raw = call_openrouter(
        build_payload(prompt, structured=False, response_model=response_model), api_key
    )
    return parse_response(raw, response_model)


def merge_local_and_llm(
    local_candidates: dict[str, Candidate],
    llm_result: ContractExtraction,
    requested_fields: list[str],
) -> dict[str, Candidate]:
    """Merge results while keeping local deterministic values in control."""

    merged = dict(local_candidates)
    for field in ALL_FIELDS:
        if is_locked_candidate(field, merged.get(field)):
            continue
        llm_candidate = candidate_from_llm(llm_result, field)
        if llm_candidate.value is not None:
            merged[field] = llm_candidate
        elif field in requested_fields and field in merged:
            merged[field] = Candidate()
    return merged


def extract_contract(contract_text: str, api_key: Optional[str]) -> ContractExtraction:
    """Orchestrate the full extraction flow.

    Order of operations:
    1. Run regex extraction for the fields that are usually explicit.
    2. Mark the "sometimes present" fields as LLM-only.
    3. If everything that should be routed is resolved, return immediately.
    4. If an API key exists, call the LLM only for unresolved or LLM-only fields.
    5. Merge the two result sources and return one final object.
    6. If the LLM fails, return the local partial result with a warning.
    """

    local_candidates, warnings = extract_regex_candidates(contract_text)
    missing_fields = get_missing_fields(local_candidates)
    if not missing_fields:
        return build_result_from_candidates(local_candidates, warnings)

    if not api_key:
        warnings.append(
            "LLM fallback skipped because OPENROUTER_API_KEY is not set; unresolved fields remain null."
        )
        return build_result_from_candidates(local_candidates, warnings)

    try:
        llm_result = call_llm_for_missing_fields(
            contract_text,
            missing_fields,
            get_locked_values(local_candidates),
            api_key,
        )
    except (requests.RequestException, ValueError, ValidationError) as exc:
        warnings.append(
            f"LLM fallback failed ({type(exc).__name__}); unresolved fields remain null."
        )
        return build_result_from_candidates(local_candidates, warnings)

    merged = merge_local_and_llm(local_candidates, llm_result, missing_fields)
    warnings.extend(llm_result.warnings)
    return build_result_from_candidates(merged, warnings)


def main() -> int:
    """Command-line entrypoint.

    It reads the contract file, runs extraction, adds elapsed time in rounded
    seconds, and prints or writes the final JSON payload.
    """

    start_time = perf_counter()

    parser = argparse.ArgumentParser(
        description="Extract contract metadata via OpenRouter using Pydantic"
    )
    parser.add_argument(
        "--contract", required=True, help="Path to UTF-8 contract text file"
    )
    parser.add_argument("--output", help="Path to write output JSON")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")

    contract_text = Path(args.contract).read_text(encoding="utf-8")

    try:
        result = extract_contract(contract_text, api_key)
    except (requests.RequestException, ValueError, ValidationError) as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        return 2

    elapsed_time_seconds = int(round(perf_counter() - start_time))
    result = result.model_copy(update={"elapsed_time_seconds": elapsed_time_seconds})

    output_json = result.model_dump_json(indent=2, exclude_none=False)
    if args.output:
        Path(args.output).write_text(output_json + "\n", encoding="utf-8")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
