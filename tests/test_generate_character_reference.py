from pathlib import Path
from unittest.mock import patch

import pytest

from generate_character_reference import build_prompt, main


def test_build_prompt_includes_style_anchor_and_neutral_pose():
    prompt = build_prompt()
    assert "Hand-drawn 2D doodle cartoon animation" in prompt
    assert "neutral pose" in prompt.lower()
    assert "white background" in prompt.lower()


def test_main_writes_to_channel_data_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "channel_data").mkdir()

    def mock_generate(prompt, output_path, **kw):
        output_path.write_bytes(b"\x89PNG-ref")
        return output_path

    with patch("generate_character_reference.ImageGenerator") as mock_gen_cls:
        mock_gen_cls.return_value.generate.side_effect = mock_generate
        result = main([])

    assert result == tmp_path / "channel_data" / "character_reference.png"
    assert result.read_bytes() == b"\x89PNG-ref"


def test_main_raises_without_gemini_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "channel_data").mkdir()

    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        main([])
