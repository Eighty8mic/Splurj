from unittest.mock import MagicMock, patch

import pytest

from engine.images import ImageGenerator, _prompt_cache_path


def test_requires_api_key():
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        ImageGenerator(api_key="")


def test_prompt_cache_path_is_stable_for_same_prompt_and_model():
    p1 = _prompt_cache_path("a doodle cat", "gemini-3.1-flash-image")
    p2 = _prompt_cache_path("a doodle cat", "gemini-3.1-flash-image")
    p3 = _prompt_cache_path("a doodle dog", "gemini-3.1-flash-image")
    assert p1 == p2
    assert p1 != p3


def _fake_image_response(data: bytes) -> MagicMock:
    fake_part = MagicMock(inline_data=MagicMock(data=data))
    return MagicMock(candidates=[MagicMock(content=MagicMock(parts=[fake_part]))])


def test_generate_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key")

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = (
            _fake_image_response(b"\x89PNG-fake-bytes-padding-000000")
        )
        out1 = tmp_path / "image_00.png"
        gen.generate("a doodle cat, flat colors", out1)
        out2 = tmp_path / "image_01.png"
        gen.generate("a doodle cat, flat colors", out2)  # identical prompt -> cache hit

    assert out1.read_bytes() == out2.read_bytes()
    assert mock_client_cls.return_value.models.generate_content.call_count == 1


def test_generate_passes_reference_image_alongside_prompt_text(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key", use_cache=False)

    ref_path = tmp_path / "character_reference.png"
    ref_path.write_bytes(b"\x89PNG-reference-bytes-padding-00")

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = (
            _fake_image_response(b"\x89PNG-fake-scene-bytes-padding-0")
        )
        gen.generate(
            "a doodle cat holding a wallet",
            tmp_path / "image_00.png",
            reference_image_path=ref_path,
        )

    call_kwargs = mock_client_cls.return_value.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]
    assert contents[0] == "a doodle cat holding a wallet"
    assert len(contents) == 2  # prompt text + one reference-image Part


def test_generate_retries_on_rate_limit_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key", use_cache=False)

    with patch("google.genai.Client") as mock_client_cls, \
         patch("engine.images.time.sleep") as mock_sleep:
        mock_client_cls.return_value.models.generate_content.side_effect = [
            RuntimeError("429 rate limit"),
            _fake_image_response(b"\x89PNG-fake-bytes-padding-000000"),
        ]
        gen.generate("a doodle cat", tmp_path / "image_00.png")

    mock_sleep.assert_called_once()


def test_generate_raises_after_exhausting_retries_on_persistent_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key", use_cache=False)

    with patch("google.genai.Client") as mock_client_cls, \
         patch("engine.images.time.sleep"):
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("429 rate limit")
        with pytest.raises(RuntimeError, match="failed after"):
            gen.generate("a doodle cat", tmp_path / "image_00.png", max_retries=2)
