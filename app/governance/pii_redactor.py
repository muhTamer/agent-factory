# app/governance/pii_redactor.py
"""
PII Redaction Engine — Privacy Preservation for RQ2

Detects and redacts personally identifiable information (PII) in text
and nested dict structures.  Returns both the redacted output and a
structured log of what was removed (for the audit trail).

Implements the ``pii_redaction`` flag already declared in PolicyPack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class RedactionRecord:
    """One instance of detected + redacted PII."""

    pii_type: str  # "email", "phone", "credit_card", "national_id"
    original_snippet: str  # first/last chars only, e.g. "j***@e***.com"
    position: int  # char offset in original text
    replacement: str  # e.g. "[EMAIL_REDACTED]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pii_type": self.pii_type,
            "original_snippet": self.original_snippet,
            "position": self.position,
            "replacement": self.replacement,
        }


# ── Pattern definitions ──────────────────────────────────────────────

# Email: standard RFC-5322 simplified
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Phone: international and local formats
# Matches: +45 12345678, +1-555-123-4567, (555) 123-4567, 555.123.4567
_PHONE_RE = re.compile(
    r"(?<!\d)"  # no digit before
    r"(?:\+?\d{1,3}[\s.-]?)?"  # optional country code
    r"(?:\(?\d{2,4}\)?[\s.-]?)"  # area code
    r"\d{3,4}[\s.-]?\d{3,5}"  # subscriber number
    r"(?!\d)"  # no digit after
)

# Credit card: 13-19 digits, optionally separated by spaces/dashes
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# National ID / SSN-like: XXX-XX-XXXX
_NATIONAL_ID_RE = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

# IBAN: 2-letter country code + 2 check digits + up to 30 alphanumeric
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?(?:[\dA-Z]{4}[\s]?){1,7}[\dA-Z]{1,4}\b")

_PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("email", _EMAIL_RE, "[EMAIL_REDACTED]"),
    ("credit_card", _CREDIT_CARD_RE, "[CARD_REDACTED]"),
    ("national_id", _NATIONAL_ID_RE, "[ID_REDACTED]"),
    ("iban", _IBAN_RE, "[IBAN_REDACTED]"),
    ("phone", _PHONE_RE, "[PHONE_REDACTED]"),
]


def _mask_snippet(text: str) -> str:
    """Create a safe snippet showing only first and last chars."""
    if len(text) <= 4:
        return "***"
    return text[0] + "***" + text[-1]


def _luhn_check(digits: str) -> bool:
    """Validate a digit string with the Luhn algorithm (credit cards)."""
    nums = [int(d) for d in digits if d.isdigit()]
    if len(nums) < 13:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class PIIRedactor:
    """Detect and redact PII from text and nested dicts."""

    def redact(self, text: str) -> Tuple[str, List[RedactionRecord]]:
        """Redact PII from a single string.

        Returns:
            (redacted_text, list_of_redaction_records)
        """
        records: List[RedactionRecord] = []
        result = text

        # Process patterns in order (email before phone to avoid partial matches)
        for pii_type, pattern, replacement in _PII_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group()

                # Credit card: validate with Luhn
                if pii_type == "credit_card":
                    digits_only = re.sub(r"[\s-]", "", raw)
                    if not _luhn_check(digits_only):
                        continue

                # National ID: skip if it looks like a regular number (no separators)
                if pii_type == "national_id":
                    if raw.isdigit() and len(raw) == 9:
                        # Could be a regular 9-digit number; only redact if it has separators
                        if "-" not in match.group() and " " not in match.group():
                            continue

                records.append(
                    RedactionRecord(
                        pii_type=pii_type,
                        original_snippet=_mask_snippet(raw),
                        position=match.start(),
                        replacement=replacement,
                    )
                )
                result = result.replace(raw, replacement, 1)

        return result, records

    def redact_dict(
        self, data: Dict[str, Any], _depth: int = 0
    ) -> Tuple[Dict[str, Any], List[RedactionRecord]]:
        """Deep-redact all string values in a dict (max depth 10).

        Returns:
            (redacted_dict, all_redaction_records)
        """
        if _depth > 10:
            return data, []

        all_records: List[RedactionRecord] = []
        out: Dict[str, Any] = {}

        for key, val in data.items():
            if isinstance(val, str):
                redacted, records = self.redact(val)
                out[key] = redacted
                all_records.extend(records)
            elif isinstance(val, dict):
                redacted_d, records = self.redact_dict(val, _depth + 1)
                out[key] = redacted_d
                all_records.extend(records)
            elif isinstance(val, list):
                new_list = []
                for item in val:
                    if isinstance(item, str):
                        r, recs = self.redact(item)
                        new_list.append(r)
                        all_records.extend(recs)
                    elif isinstance(item, dict):
                        r, recs = self.redact_dict(item, _depth + 1)
                        new_list.append(r)
                        all_records.extend(recs)
                    else:
                        new_list.append(item)
                out[key] = new_list
            else:
                out[key] = val

        return out, all_records
