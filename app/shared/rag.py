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
    Simple, robust FAQ RAG-style agent generator.

    - Writes config.yaml with docs list.
    - Generated agent loads docs at runtime.
    - Answers questions by naive string similarity over (question, answer) pairs.
    - Exposes rich metadata for LLM router.
    """
    gen_dir.mkdir(parents=True, exist_ok=True)

    docs = inputs.get("docs") or []
    if isinstance(docs, str):
        docs = [docs]

    cfg = {
        "id": agent_id,
        "docs": docs,
    }
    (gen_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    header = f"# Auto-generated FAQ RAG agent ({agent_id})\n"

    body = textwrap.dedent(
        '''\
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import csv, json, math, yaml
from app.runtime.interfaces import IAgent


class Agent(IAgent):
    def __init__(self) -> None:
        self.ready = False
        self.cfg: Dict[str, Any] | None = None
        self.faqs: List[Tuple[str, str]] = []  # (question, answer)

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

    def _load_faqs_from_csv(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # try common column names
                q_cols = ["question", "Question", "Q"]
                a_cols = ["answer", "Answer", "A"]
                for row in reader:
                    q = None
                    a = None
                    for qc in q_cols:
                        if qc in row and row[qc]:
                            q = row[qc].strip()
                            break
                    for ac in a_cols:
                        if ac in row and row[ac]:
                            a = row[ac].strip()
                            break
                    if q and a:
                        self.faqs.append((q, a))
        except Exception:
            # ignore bad CSV
            return

    def _load_faqs(self) -> None:
        docs = self.cfg.get("docs", []) if self.cfg else []
        for d in docs:
            p = Path(d)
            if p.suffix.lower() == ".csv":
                self._load_faqs_from_csv(p)
            # Here we could add logic for .md, .txt, etc.

    def load(self, spec: Dict[str, Any]) -> None:
        self._load_config()
        self._load_faqs()
        self.ready = True

    def _similarity(self, q: str, cand: str) -> float:
        """
        Very naive similarity: Jaccard over lowercased word sets.
        Enough for POC and deterministic.
        """
        qs = set(q.lower().split())
        cs = set(cand.lower().split())
        if not qs or not cs:
            return 0.0
        inter = len(qs & cs)
        union = len(qs | cs)
        return inter / union if union else 0.0

    def _search(self, query: str) -> Dict[str, Any]:
        if not self.faqs:
            return {"answer": "I don’t have that yet.", "score": 0.0, "citations": []}

        best_q = None
        best_a = None
        best_score = 0.0
        for q, a in self.faqs:
            s = self._similarity(query, q)
            if s > best_score:
                best_score = s
                best_q = q
                best_a = a

        if best_score <= 0.0 or best_a is None:
            return {"answer": "I don’t have that yet.", "score": 0.0, "citations": []}

        return {
            "answer": best_a,
            "score": float(best_score),
            "citations": [{"question": best_q}] if best_q else [],
        }

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        text = (request.get("text") or request.get("query") or "").strip()
        if not text:
            return {"answer": "Please provide a question.", "score": 0.0, "citations": []}

        res = self._search(text)
        return {
            "intent": "faq",
            "answer": res["answer"],
            "score": res["score"],
            "citations": res["citations"],
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "__AGENT_ID__",
            "type": "faq_rag",
            "ready": self.ready,
            "description": (
                "Answers customer FAQs using retrieval-augmented generation-style lookup "
                "over uploaded FAQ and policy-like CSV documents."
            ),
            "capabilities": [
                "faq_answering",
                "policy_lookup",
                "knowledge_base_search",
            ],
            "vertical": "generic_customer_service",
            "docs": len(self.faqs),
        }
'''
    )

    agent_src = header + body.replace("__AGENT_ID__", agent_id)
    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8")
    return gen_dir
