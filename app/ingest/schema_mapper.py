# app/ingest/schema_mapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re
import math


@dataclass
class ColumnMapResult:
    question_col: str
    answer_col: str
    confidence: float
    reasoning: str
    used_llm: bool = False


class SchemaMapper:
    """
    Detect which CSV columns correspond to "question" and "answer".

    Full-fledged behavior (per your requirement):
      - ALWAYS run LLM mapping when llm_client is available (not just fallback).
      - ALSO compute heuristic mapping (header + sample-based).
      - Compare both and decide deterministically.
      - Keep privacy: send headers always, and only tiny sanitized samples (2 rows) unless disabled.

    Notes:
      - Expects llm_client.chat_json(messages=[...], model=..., temperature=...) API.
      - DO NOT pass temperature=0.0 (some models reject it).
      - Default model is "gpt-5o-mini" (your constraint).
    """

    # Basic PII scrubbing
    _EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
    _PHONE = re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b")
    _LONG_DIGITS = re.compile(r"\b\d{6,}\b")
    _CARDISH = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    _WS = re.compile(r"\s+")
    _WORD = re.compile(r"[A-Za-z0-9]+")

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = "gpt-5o-mini",
        allow_llm: bool = True,
        allow_samples_to_llm: bool = True,
        sample_rows_for_llm: int = 2,
        max_sample_chars: int = 120,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.allow_llm = allow_llm
        self.allow_samples_to_llm = allow_samples_to_llm
        self.sample_rows_for_llm = max(0, int(sample_rows_for_llm))
        self.max_sample_chars = max(20, int(max_sample_chars))

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def map_columns(
        self,
        headers: Sequence[str],
        sample_rows: Sequence[Dict[str, str]],
    ) -> Optional[ColumnMapResult]:
        """
        Always run BOTH:
          1) Heuristics (local)
          2) LLM mapping (when available + allowed)

        Then compare and decide.
        """
        headers = [h for h in headers if isinstance(h, str) and h.strip()]
        if len(headers) < 2:
            return None

        heur = self._heuristic_proposal(headers, sample_rows)

        llm_res: Optional[ColumnMapResult] = None
        if self.allow_llm and self.llm_client and getattr(self.llm_client, "chat_json", None):
            llm_res = self._llm_map(headers, sample_rows)

        # If LLM isn't available, return heuristics
        if llm_res is None:
            return heur

        # If heuristics absent, accept LLM
        if heur is None:
            return llm_res

        # Compare & decide
        return self._combine_decisions(heur, llm_res, headers)

    # ---------------------------------------------------------------------
    # Heuristics
    # ---------------------------------------------------------------------
    def _heuristic_proposal(
        self,
        headers: Sequence[str],
        sample_rows: Sequence[Dict[str, str]],
    ) -> Optional[ColumnMapResult]:
        q_candidates = self._rank_by_header(headers, kind="question")
        a_candidates = self._rank_by_header(headers, kind="answer")

        # Strong header match shortcut
        if q_candidates and a_candidates:
            q1, qs1 = q_candidates[0]
            a1, as1 = a_candidates[0]
            if q1 != a1 and (qs1 >= 0.95 or as1 >= 0.95):
                return ColumnMapResult(
                    question_col=q1,
                    answer_col=a1,
                    confidence=min(0.98, (qs1 + as1) / 2),
                    reasoning="Heuristic: high-confidence header keyword match.",
                    used_llm=False,
                )

        scored = self._rank_by_samples(headers, sample_rows)
        if scored:
            q_col, a_col, conf, why = scored
            return ColumnMapResult(
                question_col=q_col,
                answer_col=a_col,
                confidence=float(max(0.0, min(1.0, conf))),
                reasoning="Heuristic: " + why,
                used_llm=False,
            )

        return None

    def _rank_by_header(self, headers: Sequence[str], kind: str) -> List[Tuple[str, float]]:
        kind = kind.lower()
        scores: Dict[str, float] = {h: 0.0 for h in headers}

        q_keys = [
            "question",
            "q",
            "prompt",
            "query",
            "faq_question",
            "user_query",
            "title",
            "utterance",
            "input",
        ]
        a_keys = [
            "answer",
            "a",
            "response",
            "reply",
            "completion",
            "faq_answer",
            "bot_response",
            "content",
            "text",
            "output",
        ]
        keys = q_keys if kind == "question" else a_keys

        for h in headers:
            hl = h.strip().lower()

            for k in keys:
                if hl == k:
                    scores[h] = max(scores[h], 1.0)
                elif k in hl:
                    scores[h] = max(scores[h], 0.85)

            # mild hints
            if kind == "question" and ("?" in hl or "ques" in hl):
                scores[h] = max(scores[h], 0.70)
            if kind == "answer" and ("ans" in hl or "resp" in hl):
                scores[h] = max(scores[h], 0.70)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(h, s) for (h, s) in ranked if s > 0.0]

    def _rank_by_samples(
        self,
        headers: Sequence[str],
        sample_rows: Sequence[Dict[str, str]],
    ) -> Optional[Tuple[str, str, float, str]]:
        if not sample_rows:
            return None

        stats: Dict[str, Dict[str, float]] = {}
        for h in headers:
            vals = [(r.get(h) or "").strip() for r in sample_rows]
            vals = [v for v in vals if v]
            if not vals:
                continue

            qmark_rate = sum(1 for v in vals if "?" in v) / max(1, len(vals))
            avg_len = sum(len(v) for v in vals) / max(1, len(vals))
            qw_rate = sum(1 for v in vals if self._looks_like_question(v)) / max(1, len(vals))

            stats[h] = {
                "qmark_rate": float(qmark_rate),
                "avg_len": float(avg_len),
                "qw_rate": float(qw_rate),
            }

        if len(stats) < 2:
            return None

        best: Optional[Tuple[str, str, float, str]] = None

        for q in headers:
            if q not in stats:
                continue
            for a in headers:
                if a == q or a not in stats:
                    continue

                qs = stats[q]
                ans = stats[a]

                # question-ness
                q_score = 0.55 * qs["qmark_rate"] + 0.45 * qs["qw_rate"]

                # answer tends to be longer (sigmoid-ish)
                len_diff = ans["avg_len"] - qs["avg_len"]
                a_score = 1 / (1 + math.exp(-0.02 * len_diff))

                conf = 0.55 * q_score + 0.45 * a_score

                if best is None or conf > best[2]:
                    why = (
                        f"qmark_rate={qs['qmark_rate']:.2f}, qw_rate={qs['qw_rate']:.2f} for '{q}', "
                        f"avg_len(q)={qs['avg_len']:.1f}, avg_len(a)={ans['avg_len']:.1f}."
                    )
                    best = (q, a, float(conf), why)

        return best

    def _looks_like_question(self, s: str) -> bool:
        s = (s or "").strip().lower()
        if not s:
            return False
        starters = (
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "can",
            "do",
            "does",
            "is",
            "are",
            "will",
            "could",
            "should",
            "may",
        )
        return s.endswith("?") or s.startswith(starters)

    # ---------------------------------------------------------------------
    # Combine decisions
    # ---------------------------------------------------------------------
    def _combine_decisions(
        self,
        heur: ColumnMapResult,
        llm_res: ColumnMapResult,
        headers: Sequence[str],
    ) -> ColumnMapResult:
        h_pair = (heur.question_col, heur.answer_col)
        l_pair = (llm_res.question_col, llm_res.answer_col)

        # sanity
        if (
            heur.question_col not in headers
            or heur.answer_col not in headers
            or heur.question_col == heur.answer_col
        ):
            return llm_res
        if (
            llm_res.question_col not in headers
            or llm_res.answer_col not in headers
            or llm_res.question_col == llm_res.answer_col
        ):
            return heur

        if h_pair == l_pair:
            conf = min(0.99, max(heur.confidence, llm_res.confidence) + 0.05)
            return ColumnMapResult(
                question_col=h_pair[0],
                answer_col=h_pair[1],
                confidence=conf,
                reasoning=(
                    "Heuristics and LLM agree. "
                    f"Heur={heur.confidence:.2f}, LLM={llm_res.confidence:.2f}. "
                    f"HeurReason='{heur.reasoning}' LLMReason='{llm_res.reasoning}'"
                )[:600],
                used_llm=True,
            )

        h = float(heur.confidence)
        llm = float(llm_res.confidence)

        if llm >= h + 0.15:
            return ColumnMapResult(
                question_col=llm_res.question_col,
                answer_col=llm_res.answer_col,
                confidence=llm,
                reasoning=(
                    "LLM chosen over heuristics due to higher confidence. "
                    f"LLM={llm:.2f} vs Heur={h:.2f}. "
                    f"LLMReason='{llm_res.reasoning}' HeurReason='{heur.reasoning}'"
                )[:600],
                used_llm=True,
            )

        if h >= llm + 0.15:
            return ColumnMapResult(
                question_col=heur.question_col,
                answer_col=heur.answer_col,
                confidence=h,
                reasoning=(
                    "Heuristics chosen over LLM due to higher confidence. "
                    f"Heur={h:.2f} vs LLM={llm:.2f}. "
                    f"HeurReason='{heur.reasoning}' LLMReason='{llm_res.reasoning}'"
                )[:600],
                used_llm=True,  # LLM still ran
            )

        # Close call: default to LLM but reduce confidence
        conf = max(0.35, min(0.85, llm - 0.10))
        return ColumnMapResult(
            question_col=llm_res.question_col,
            answer_col=llm_res.answer_col,
            confidence=conf,
            reasoning=(
                "LLM and heuristics disagree with similar confidence; defaulting to LLM "
                "but marking uncertainty. "
                f"LLM={llm:.2f} vs Heur={h:.2f}. "
                f"LLMReason='{llm_res.reasoning}' HeurReason='{heur.reasoning}'"
            )[:600],
            used_llm=True,
        )

    # ---------------------------------------------------------------------
    # LLM mapping (always run when available)
    # ---------------------------------------------------------------------
    def _sanitize(self, s: str) -> str:
        s = (s or "").strip()
        s = self._EMAIL.sub("[email]", s)
        s = self._CARDISH.sub("[card]", s)
        s = self._LONG_DIGITS.sub("[id]", s)
        s = self._PHONE.sub("[phone]", s)
        s = self._WS.sub(" ", s).strip()
        if len(s) > self.max_sample_chars:
            s = s[: self.max_sample_chars - 1] + "…"
        return s

    def _llm_map(
        self,
        headers: Sequence[str],
        sample_rows: Sequence[Dict[str, str]],
    ) -> Optional[ColumnMapResult]:
        preview: Dict[str, List[str]] = {h: [] for h in headers}

        if self.allow_samples_to_llm and self.sample_rows_for_llm > 0:
            for r in sample_rows[: self.sample_rows_for_llm]:
                for h in headers:
                    v = (r.get(h) or "").strip()
                    if v:
                        preview[h].append(self._sanitize(v))

        system = (
            "You map CSV columns to an FAQ schema.\n"
            "Given column headers and a few sanitized examples, identify which column is the question "
            "and which is the answer.\n"
            "Return ONLY JSON with keys: question_col, answer_col, confidence, reasoning.\n"
            "confidence must be 0..1.\n"
            "Do not invent column names.\n"
        )

        payload = {
            "headers": list(headers),
            "examples": preview,
            "task": "Select the best question_col and answer_col for an FAQ dataset.",
        }

        try:
            res = self.llm_client.chat_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(payload)},
                ],
                model=self.model,
                # do NOT pass temperature=0.0; keep default or 1.0
            )
        except Exception as e:
            # If LLM fails, return None so caller can rely on heuristics
            return ColumnMapResult(
                question_col=headers[0],
                answer_col=headers[1],
                confidence=0.25,
                reasoning=f"LLM mapping failed; fallback to first two columns. Error: {e}",
                used_llm=True,
            )

        q = str(res.get("question_col") or "").strip()
        a = str(res.get("answer_col") or "").strip()
        why = str(res.get("reasoning") or "").strip()

        if q not in headers or a not in headers or q == a:
            return None

        try:
            conf = float(res.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        return ColumnMapResult(
            question_col=q,
            answer_col=a,
            confidence=conf,
            reasoning=why or "LLM-selected mapping.",
            used_llm=True,
        )
