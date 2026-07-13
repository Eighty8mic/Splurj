"""
Automatic daily topic selection for Splurj.

Picks a citation from the approved (non-contested) citation bank and pairs
it with one of the channel's five proven viral-angle formulas, avoiding any
(citation, angle) combination already recorded in used_topics.json.
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List

ANGLES = {
    "why_brain": (
        "Why your brain ___ your money — evolutionary mismatch: a brain built for scarcity "
        "operating in a world of one-click spending"
    ),
    "cant_stop": (
        "The real reason you can't ___ — explains a universal money struggle through named research"
    ),
    "designed_to": (
        "___ was designed to make you ___ — reveals deliberate engineering inside a financial "
        "system or product"
    ),
    "the_effect": (
        "The ___ Effect — names a real experiment and mirrors its conclusion onto the viewer's "
        "daily spending"
    ),
    "never_noticed": (
        "You never noticed that ___ — exposes a hidden mechanic inside something the viewer pays "
        "for every day"
    ),
}


def load_citations(path: Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_used_topics(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def record_used_topic(path: Path, entry: Dict[str, Any]) -> None:
    path = Path(path)
    entries = load_used_topics(path)
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def pick_topic(citations: List[Dict[str, Any]], used_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    approved = {c["id"]: c for c in citations if c["status"] == "approved" and not c.get("contested")}

    used_pairs = {
        (u["citation_ids"][0], u["angle_key"])
        for u in used_topics
        if len(u["citation_ids"]) == 1 and u["citation_ids"][0] in approved
    }

    candidates = [
        (citation_id, angle_key)
        for citation_id in approved
        for angle_key in ANGLES
        if (citation_id, angle_key) not in used_pairs
    ]

    if not candidates:
        raise RuntimeError(
            "All citation/angle combinations exhausted — approve more citations in "
            "channel_data/citations.json before the next draft."
        )

    citation_id, angle_key = random.choice(candidates)
    return {
        "citation_ids": [citation_id],
        "citation": approved[citation_id],
        "angle_key": angle_key,
        "angle_formula": ANGLES[angle_key],
    }
