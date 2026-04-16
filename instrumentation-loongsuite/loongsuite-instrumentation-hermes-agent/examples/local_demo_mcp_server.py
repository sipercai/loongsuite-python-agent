from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP


KB_PATH = Path(
    os.environ.get(
        "HERMES_DEMO_KB_PATH",
        Path(__file__).with_name("demo_knowledge_base.json"),
    )
)

SERVER = FastMCP("demo")

_STOPWORDS = {
    "the",
    "with",
    "under",
    "and",
    "for",
    "from",
    "that",
    "this",
    "into",
    "just",
    "without",
}


def _load_knowledge_base() -> list[dict]:
    data = json.loads(KB_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Knowledge base must be a list of documents")
    return data


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@SERVER.tool(description="Return the current time for a given IANA timezone.")
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    current = datetime.now(ZoneInfo(timezone))
    return json.dumps(
        {
            "timezone": timezone,
            "iso": current.isoformat(),
            "weekday": current.strftime("%A"),
        },
        ensure_ascii=False,
    )


@SERVER.tool(description="Search the local demo knowledge base and return a grounded answer with evidence.")
def search_briefing(query: str) -> str:
    query_lower = _normalize(query)
    docs = _load_knowledge_base()
    for doc in docs:
        haystacks = [
            _normalize(doc.get("topic", "")),
            _normalize(doc.get("answer", "")),
            _normalize(doc.get("evidence", "")),
            " ".join(_normalize(alias) for alias in doc.get("aliases", [])),
        ]
        if any(query_lower in haystack for haystack in haystacks):
            return json.dumps(
                {
                    "query": query,
                    "source_id": doc["id"],
                    "answer": doc["answer"],
                    "evidence": doc["evidence"],
                },
                ensure_ascii=False,
            )
    return json.dumps(
        {
            "query": query,
            "source_id": "",
            "answer": "NOT_FOUND",
            "evidence": "No matching grounding snippet was found.",
        },
        ensure_ascii=False,
    )


@SERVER.tool(description="Grade a candidate answer against a reference answer using simple keyword overlap.")
def grade_candidate(
    reference: str,
    candidate: str,
    rubric: str = "exact_keyword_overlap",
) -> str:
    keywords = sorted(
        {
            token
            for token in re.findall(r"[A-Za-z0-9_>+-]+", reference.lower())
            if len(token) > 2 and token not in _STOPWORDS
        }
    )
    candidate_lower = candidate.lower()
    matched = [token for token in keywords if token in candidate_lower]
    score = round(len(matched) / len(keywords), 4) if keywords else 1.0
    verdict = "PASS" if score >= 0.75 else "FAIL"
    return json.dumps(
        {
            "rubric": rubric,
            "verdict": verdict,
            "score": score,
            "matched_keywords": matched,
            "missing_keywords": [token for token in keywords if token not in matched],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    SERVER.run("stdio")
