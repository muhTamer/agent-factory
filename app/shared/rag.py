# app/shared/rag.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import csv
import math
import re
import yaml
import textwrap

# ---------------------------
# Simple, dependency-free RAG
# ---------------------------

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s or "")]


def _read_text(path: Path, limit: int | None = None) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if limit:
            return txt[:limit]
        return txt
    except Exception:
        return ""


def _markdown_chunks(md: str, max_chars: int = 1200) -> List[str]:
    # naive: split by headings / paragraphs, then re-chunk
    parts = re.split(r"\n\s*#+\s", md)
    chunks, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


@dataclass
class CorpusItem:
    text: str
    source: str  # filename
    kind: str  # "csv_qa" | "md" | "txt" | "other"
    meta: Dict[str, Any]


@dataclass
class Index:
    items: List[CorpusItem]
    vocab: Dict[str, int]  # token -> df
    vecs: List[Dict[str, float]]  # tf-idf sparse vector per item
    idf: Dict[str, float]


# ---------------------------
# Loading documents
# ---------------------------


def load_corpus(paths: List[str]) -> List[CorpusItem]:
    items: List[CorpusItem] = []
    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        name = path.name
        ext = path.suffix.lower()

        if ext == ".csv":
            # Try to load Q/A rows
            try:
                with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
                    reader = csv.DictReader(fh)
                    cols = [c.lower() for c in (reader.fieldnames or [])]
                    q_col = next((c for c in cols if c in {"q", "question"}), None)
                    a_col = next((c for c in cols if c in {"a", "answer"}), None)
                    if q_col and a_col:
                        for row in reader:
                            q = row.get(q_col, "").strip()
                            a = row.get(a_col, "").strip()
                            if q and a:
                                items.append(
                                    CorpusItem(
                                        text=f"Q: {q}\nA: {a}",
                                        source=name,
                                        kind="csv_qa",
                                        meta={"q": q, "a": a},
                                    )
                                )
                    else:
                        # treat whole file as text
                        txt = _read_text(path, limit=20000)
                        for ch in _markdown_chunks(txt):
                            items.append(CorpusItem(ch, name, "txt", {}))
            except Exception:
                txt = _read_text(path, limit=20000)
                for ch in _markdown_chunks(txt):
                    items.append(CorpusItem(ch, name, "txt", {}))

        elif ext in {".md", ".txt"}:
            md = _read_text(path, limit=40000)
            for ch in _markdown_chunks(md):
                items.append(CorpusItem(ch, name, "md", {}))

        else:
            # yaml/json etc. → index as text snapshot
            txt = _read_text(path, limit=12000)
            if txt:
                for ch in _markdown_chunks(txt, max_chars=1000):
                    items.append(CorpusItem(ch, name, "other", {}))
    return items


# ---------------------------
# Index building (TF-IDF-ish)
# ---------------------------


def build_index(items: List[CorpusItem]) -> Index:
    # doc freq
    df: Dict[str, int] = {}
    docs_tokens: List[List[str]] = []
    for it in items:
        toks = list(dict.fromkeys(_tok(it.text)))  # unique per doc
        docs_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1

    N = max(1, len(items))
    idf = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

    # tf-idf vecs
    vecs: List[Dict[str, float]] = []
    for it in items:
        tf: Dict[str, int] = {}
        for t in _tok(it.text):
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        norm = 0.0
        for t, f in tf.items():
            w = (1 + math.log(f)) * idf.get(t, 0.0)
            vec[t] = w
            norm += w * w
        norm = math.sqrt(max(1e-9, norm))
        for t in list(vec.keys()):
            vec[t] /= norm
        vecs.append(vec)

    return Index(items=items, vocab=df, vecs=vecs, idf=idf)


def _cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    # iterate smaller
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    s = 0.0
    for t, w in vec_a.items():
        w2 = vec_b.get(t)
        if w2 is not None:
            s += w * w2
    return float(max(0.0, min(1.0, s)))


def query_index(index: Index, query: str, k: int = 5) -> List[Tuple[float, CorpusItem]]:
    tf: Dict[str, int] = {}
    for t in _tok(query):
        tf[t] = tf.get(t, 0) + 1
    vec: Dict[str, float] = {}
    norm = 0.0
    for t, f in tf.items():
        w = (1 + math.log(f)) * index.idf.get(t, 0.0)
        vec[t] = w
        norm += w * w
    norm = math.sqrt(max(1e-9, norm))
    for t in list(vec.keys()):
        vec[t] /= norm

    scores: List[Tuple[float, CorpusItem]] = []
    for v, it in zip(index.vecs, index.items):
        s = _cosine(vec, v)
        scores.append((s, it))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]


# ---------------------------
# Answer synthesis
# ---------------------------


def exact_faq_answer(items: List[CorpusItem], query: str) -> Tuple[str | None, Dict[str, Any]]:
    q_norm = " ".join(_tok(query))
    best: Tuple[int, str, str] | None = None  # (len_dist, a, src)
    for it in items:
        if it.kind != "csv_qa":
            continue
        q2 = " ".join(_tok(it.meta.get("q", "")))
        if not q2:
            continue
        # simple equality or high overlap
        if q2 == q_norm:
            return it.meta.get("a"), {"source": it.source, "match": "exact"}
        # track nearest length/overlap (quick heuristic)
        dist = abs(len(q2) - len(q_norm))
        cand = (dist, it.meta.get("a", ""), it.source)
        if best is None or cand < best:
            best = cand
    if best:
        return best[1], {"source": best[2], "match": "near"}
    return None, {}


def synthesize_answer(query: str, hits: List[Tuple[float, CorpusItem]]) -> Dict[str, Any]:
    if not hits:
        return {"answer": "I don’t have that yet.", "citations": [], "score": 0.0}
    top_score, top = hits[0]
    # extractive-ish: return the top chunk (first ~450 chars)
    snippet = textwrap.shorten(top.text.replace("\n", " ").strip(), width=450, placeholder=" ...")
    return {
        "answer": snippet,
        "citations": [{"source": h[1].source, "score": round(float(h[0]), 3)} for h in hits[:3]],
        "score": round(float(top_score), 3),
    }


# ---------------------------
# Generator entrypoint
# ---------------------------
def build_agent(agent_id: str, inputs: dict, gen_dir: Path) -> Path:
    """
    FAQ RAG-style agent generator (dependency-free, but can use LLM for schema mapping).

    What it does:
      - Writes config.yaml with docs list
      - For each CSV:
          * Uses SchemaMapper to map columns -> question/answer
            - ALWAYS runs LLM mapping if available, and compares to heuristics
          * Extracts (q,a) pairs into a normalized faqs.json beside the agent
      - Generated agent loads faqs.json (not raw CSV) and builds TF-IDF index
      - Answers questions with TF-IDF cosine similarity + relevance gate
      - Exposes metadata for router
    """
    import json
    import textwrap
    import csv
    from pathlib import Path

    from app.ingest.schema_mapper import SchemaMapper

    gen_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # inputs
    # ---------------------------
    docs = inputs.get("docs") or []
    if isinstance(docs, str):
        docs = [docs]

    # Write config for runtime inspection/debugging
    cfg = {"id": agent_id, "docs": docs}
    (gen_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    # ---------------------------
    # LLM adapter (optional)
    # ---------------------------
    llm_adapter = None
    try:
        # Your app/llm_client.py exposes chat_json(messages, model, temperature)
        from app.llm_client import chat_json

        class _LLMAdapter:
            def chat_json(self, messages, model=None, temperature=1.0):
                # IMPORTANT: do NOT set temperature=0.0 (some models reject it)
                return chat_json(messages=messages, model=model, temperature=temperature)

        llm_adapter = _LLMAdapter()
    except Exception:
        llm_adapter = None

    mapper = SchemaMapper(
        llm_client=llm_adapter,
        model="gpt-5o-mini",  # your requirement
        allow_llm=True,
        allow_samples_to_llm=True,  # privacy: sends only tiny sanitized samples
        sample_rows_for_llm=2,
        max_sample_chars=120,
    )

    # ---------------------------
    # Normalize FAQ pairs into faqs.json
    # ---------------------------
    faqs: List[Dict[str, Any]] = []
    mapping_debug: List[Dict[str, Any]] = []

    def _read_csv_sample(path: Path, max_rows: int = 20) -> Tuple[List[str], List[Dict[str, str]]]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                headers = list(reader.fieldnames or [])
                rows: List[Dict[str, str]] = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    # keep only string-ish values
                    rows.append({k: (row.get(k) or "") for k in headers})
                return headers, rows
        except Exception:
            return [], []

    def _extract_qas(path: Path, q_col: str, a_col: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return out
                for row in reader:
                    q = (row.get(q_col) or "").strip()
                    a = (row.get(a_col) or "").strip()
                    if q and a:
                        out.append((q, a))
        except Exception:
            return out
        return out

    for d in docs:
        p = Path(d)
        if not p.exists() or not p.is_file():
            continue

        if p.suffix.lower() != ".csv":
            # For now we only normalize CSV Q/A into faqs.json.
            # (MD/YAML can be handled later as "kb chunks" blueprint if you want.)
            continue

        headers, sample_rows = _read_csv_sample(p, max_rows=25)
        if len(headers) < 2:
            mapping_debug.append(
                {
                    "file": p.name,
                    "status": "skipped",
                    "reason": "csv_missing_headers",
                }
            )
            continue

        res = mapper.map_columns(headers=headers, sample_rows=sample_rows)
        if not res:
            mapping_debug.append(
                {
                    "file": p.name,
                    "status": "failed",
                    "reason": "no_mapping_found",
                    "headers": headers,
                }
            )
            continue

        q_col = res.question_col
        a_col = res.answer_col

        mapping_debug.append(
            {
                "file": p.name,
                "status": "mapped",
                "question_col": q_col,
                "answer_col": a_col,
                "confidence": res.confidence,
                "used_llm": res.used_llm,
                "reasoning": res.reasoning,
            }
        )

        pairs = _extract_qas(p, q_col=q_col, a_col=a_col)
        for q, a in pairs:
            faqs.append(
                {
                    "q": q,
                    "a": a,
                    "source": p.name,
                }
            )

    # Write normalized data + mapping debug
    (gen_dir / "faqs.json").write_text(
        json.dumps(faqs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (gen_dir / "mapping_debug.json").write_text(
        json.dumps(mapping_debug, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------------------
    # Generate agent.py (loads faqs.json, TF-IDF retrieval)
    # ---------------------------
    header = f"# Auto-generated FAQ RAG agent ({agent_id})\n"
    body = textwrap.dedent(
        """\
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json, math, re, yaml
from app.runtime.interfaces import IAgent

_WORD = re.compile(r"[A-Za-z0-9]+")

def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s or "")]

def _build_tfidf(items: List[str]):
    df = {}
    for text in items:
        toks = list(dict.fromkeys(_tok(text)))
        for t in toks:
            df[t] = df.get(t, 0) + 1

    N = max(1, len(items))
    idf = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

    vecs = []
    for text in items:
        tf = {}
        for t in _tok(text):
            tf[t] = tf.get(t, 0) + 1
        vec = {}
        norm = 0.0
        for t, f in tf.items():
            w = (1 + math.log(f)) * idf.get(t, 0.0)
            vec[t] = w
            norm += w * w
        norm = math.sqrt(max(1e-9, norm))
        for t in list(vec.keys()):
            vec[t] /= norm
        vecs.append(vec)

    return idf, vecs

class Agent(IAgent):
    def __init__(self) -> None:
        self.ready = False
        self.cfg: Dict[str, Any] | None = None

        # normalized faqs: list[dict(q,a,source)]
        self.faqs: List[Dict[str, Any]] = []

        # search index over combined Q+A
        self._idf: Dict[str, float] = {}
        self._vecs: List[Dict[str, float]] = []
        self._texts: List[str] = []  # aligned to faqs

        # Thread-aware RAG FSM engines
        self._fsm_engines: Dict[str, Any] = {}
        self._memory: Optional[object] = None

    def _load_config(self) -> None:
        cfg_path = Path(__file__).parent / "config.yaml"
        if not cfg_path.exists():
            self.cfg = {"id": "__AGENT_ID__", "docs": []}
        else:
            self.cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if "id" not in self.cfg:
                self.cfg["id"] = "__AGENT_ID__"
            if "docs" not in self.cfg:
                self.cfg["docs"] = []

    def _load_faqs(self) -> None:
        data_path = Path(__file__).parent / "faqs.json"
        if not data_path.exists():
            self.faqs = []
            return
        try:
            self.faqs = json.loads(data_path.read_text(encoding="utf-8")) or []
            if not isinstance(self.faqs, list):
                self.faqs = []
        except Exception:
            self.faqs = []

    def _build_index(self) -> None:
        self._texts = []
        for it in self.faqs:
            q = str(it.get("q", "")).strip()
            a = str(it.get("a", "")).strip()
            if q and a:
                self._texts.append("Q: " + q + " A: " + a)

        if not self._texts:
            self._idf, self._vecs = {}, []
            return

        self._idf, self._vecs = _build_tfidf(self._texts)

    def load(self, spec: Dict[str, Any]) -> None:
        self._load_config()
        self._load_faqs()
        self._build_index()

        # Try to load conversation memory (optional)
        try:
            from app.runtime.memory import ConversationMemory
            self._memory = ConversationMemory()
        except ImportError:
            self._memory = None

        self.ready = True

    def _get_thread_id(self, request: Dict[str, Any]) -> str:
        ctx = request.get("context") if isinstance(request, dict) else None
        if isinstance(ctx, dict) and ctx.get("thread_id"):
            return str(ctx["thread_id"])
        if isinstance(request, dict) and request.get("thread_id"):
            return str(request["thread_id"])
        return "default"

    def _fsm_for(self, thread_id: str):
        if thread_id not in self._fsm_engines:
            try:
                from app.runtime.rag_fsm import RAGFiniteStateMachine, RAGFSMConfig
                self._fsm_engines[thread_id] = RAGFiniteStateMachine(
                    agent_id="__AGENT_ID__",
                    faqs=self.faqs,
                    idf=self._idf,
                    vecs=self._vecs,
                    texts=self._texts,
                    config=RAGFSMConfig(),
                    memory=self._memory,
                )
            except ImportError:
                return None
        return self._fsm_engines[thread_id]

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        text = (request.get("text") or request.get("query") or "").strip()
        if not text:
            return {"intent": "faq", "answer": "Please provide a question.", "score": 0.0, "citations": []}

        thread_id = self._get_thread_id(request)
        fsm = self._fsm_for(thread_id)

        if fsm is None:
            # Fallback: simple TF-IDF search if rag_fsm unavailable
            return self._search_fallback(text)

        result = fsm.step(query=text, thread_id=thread_id)

        response: Dict[str, Any] = {
            "intent": "faq",
            "answer": result.answer or "",
            "score": result.score,
            "citations": result.citations,
            "rag_state": result.state.value,
        }

        # Delegation signal (spine reads this)
        if result.delegation_target:
            response["delegation_target"] = result.delegation_target
            response["delegation_reason"] = result.delegation_reason

        # Clarification signal (spine reads this for pinning)
        if result.clarification_question:
            response["rag_clarification"] = True
            response["answer"] = result.clarification_question

        if result.state.value == "respond":
            response["rag_answered"] = True

        # Solvability metadata for thesis tracing
        if result.solvability:
            response["solvability"] = {
                "tfidf_score": result.solvability.tfidf_score,
                "coverage_ratio": result.solvability.coverage_ratio,
                "confidence": result.solvability.confidence,
                "should_delegate": result.solvability.should_delegate,
                "reasoning": result.solvability.reasoning,
            }

        return response

    def _search_fallback(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        \"\"\"Fallback TF-IDF search without FSM (backward compatibility).\"\"\"
        if not self._texts or not self._vecs:
            return {"intent": "faq", "answer": "I don't have that yet.", "score": 0.0, "citations": []}

        from app.runtime.rag_fsm import _vec_query, _cosine
        qv = _vec_query(query, self._idf)
        scored = []
        for i, dv in enumerate(self._vecs):
            s = _cosine(qv, dv)
            scored.append((s, i))
        scored.sort(reverse=True, key=lambda x: x[0])

        best_score, best_i = scored[0]
        if best_score < 0.12:
            return {
                "intent": "faq",
                "answer": "I couldn't find that in the provided documents.",
                "score": float(best_score),
                "citations": [],
            }

        hits = []
        for s, i in scored[:top_k]:
            it = self.faqs[i]
            hits.append({
                "score": float(s),
                "question": str(it.get("q", "")),
                "answer": str(it.get("a", "")),
                "source": str(it.get("source", "")),
            })

        return {
            "intent": "faq",
            "answer": hits[0]["answer"],
            "score": float(hits[0]["score"]),
            "citations": [{"question": h["question"], "source": h["source"]} for h in hits[:3]],
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "__AGENT_ID__",
            "type": "faq_rag",
            "ready": self.ready,
            "description": "Multi-turn FAQ RAG agent with solvability estimation, clarification, and delegation.",
            "capabilities": ["faq_answering", "policy_lookup", "knowledge_base_search",
                           "multi_turn", "clarification", "delegation"],
            "docs": len(self.faqs),
        }
"""
    )

    agent_src = header + body.replace("__AGENT_ID__", agent_id)
    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8")

    return gen_dir
