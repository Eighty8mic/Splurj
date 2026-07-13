import json

import pytest

from engine.topic_picker import (
    ANGLES,
    load_citations,
    load_used_topics,
    pick_topic,
    record_used_topic,
)


def _citation(id, status="approved", contested=False):
    return {
        "id": id, "researchers": f"Researcher {id}", "study_ref": f"Study {id}",
        "year": 2000 + id, "venue": "Journal", "summary": "summary",
        "source_url": "https://example.com", "status": status, "contested": contested,
    }


def test_load_citations_reads_json(tmp_path):
    path = tmp_path / "citations.json"
    path.write_text(json.dumps([_citation(1)]), encoding="utf-8")

    result = load_citations(path)

    assert result == [_citation(1)]


def test_load_used_topics_returns_empty_list_when_file_missing(tmp_path):
    assert load_used_topics(tmp_path / "missing.json") == []


def test_load_used_topics_reads_json(tmp_path):
    path = tmp_path / "used_topics.json"
    entry = {"citation_ids": [1], "angle_key": "why_brain", "video_day": 3}
    path.write_text(json.dumps([entry]), encoding="utf-8")

    assert load_used_topics(path) == [entry]


def test_record_used_topic_appends_to_new_file(tmp_path):
    path = tmp_path / "used_topics.json"
    record_used_topic(path, {"citation_ids": [1], "angle_key": "why_brain", "video_day": 1})
    record_used_topic(path, {"citation_ids": [2], "angle_key": "the_effect", "video_day": 2})

    result = load_used_topics(path)

    assert len(result) == 2
    assert result[0]["video_day"] == 1
    assert result[1]["video_day"] == 2


def test_pick_topic_excludes_contested_citations():
    citations = [_citation(1, contested=True), _citation(2, contested=False)]

    for _ in range(20):
        pick = pick_topic(citations, used_topics=[])
        assert pick["citation_ids"] == [2]


def test_pick_topic_excludes_non_approved_citations():
    citations = [_citation(1, status="pending"), _citation(2, status="approved")]

    for _ in range(20):
        pick = pick_topic(citations, used_topics=[])
        assert pick["citation_ids"] == [2]


def test_pick_topic_avoids_already_used_combinations():
    citations = [_citation(1)]
    # Every angle except one has already been used for citation 1.
    used = [
        {"citation_ids": [1], "angle_key": key, "video_day": i}
        for i, key in enumerate(ANGLES) if key != "the_effect"
    ]

    pick = pick_topic(citations, used_topics=used)

    assert pick["angle_key"] == "the_effect"


def test_pick_topic_raises_when_all_combinations_exhausted():
    citations = [_citation(1)]
    used = [{"citation_ids": [1], "angle_key": key, "video_day": i} for i, key in enumerate(ANGLES)]

    with pytest.raises(RuntimeError, match="exhausted"):
        pick_topic(citations, used_topics=used)


def test_pick_topic_returns_matching_citation_and_angle_formula():
    citations = [_citation(5)]

    pick = pick_topic(citations, used_topics=[])

    assert pick["citation"]["id"] == 5
    assert pick["angle_formula"] == ANGLES[pick["angle_key"]]
