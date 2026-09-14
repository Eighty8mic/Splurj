import pickle
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from engine.youtube import DEFAULT_CATEGORY, YouTubeUploader, _truncate_tags, has_valid_cached_token


def _make_uploader(tmp_path):
    with patch("engine.youtube.YouTubeUploader._authenticate", return_value=MagicMock()):
        return YouTubeUploader(client_secret_file=str(tmp_path / "secret.json"), privacy_status="private")


def test_default_category_is_education():
    assert DEFAULT_CATEGORY == "27"


def test_upload_builds_body_with_default_category_and_privacy(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_request = MagicMock()
    fake_request.next_chunk.return_value = (None, {"id": "vid123"})
    uploader.youtube.videos.return_value.insert.return_value = fake_request

    video_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake-mp4-bytes")

    response = uploader.upload(
        video_path=video_path,
        title="Why Your Brain Feels No Pain When You Tap a Card",
        description="A description with the disclaimer already appended.",
        tags=["money psychology", "splurj"],
    )

    assert response["id"] == "vid123"
    _, call_kwargs = uploader.youtube.videos.return_value.insert.call_args
    assert call_kwargs["body"]["snippet"]["categoryId"] == "27"
    assert call_kwargs["body"]["status"]["privacyStatus"] == "private"


def test_upload_retries_on_5xx_then_succeeds(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_request = MagicMock()
    fake_request.next_chunk.side_effect = [
        HttpError(resp=MagicMock(status=500), content=b"server error"),
        (None, {"id": "vid456"}),
    ]
    uploader.youtube.videos.return_value.insert.return_value = fake_request

    video_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake-mp4-bytes")

    with patch("engine.youtube.time.sleep"):
        response = uploader.upload(
            video_path=video_path, title="t", description="d", tags=[],
        )

    assert response["id"] == "vid456"


def test_upload_does_not_retry_on_programming_error(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_request = MagicMock()
    fake_request.next_chunk.side_effect = TypeError("boom: bad argument")
    uploader.youtube.videos.return_value.insert.return_value = fake_request

    video_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake-mp4-bytes")

    with patch("engine.youtube.time.sleep") as mock_sleep:
        with pytest.raises(TypeError, match="boom"):
            uploader.upload(
                video_path=video_path, title="t", description="d", tags=[],
            )

    mock_sleep.assert_not_called()


def test_set_default_thumbnail_calls_thumbnails_set(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_set = MagicMock()
    uploader.youtube.thumbnails.return_value.set.return_value = fake_set

    thumb_path = tmp_path / "thumb_00.png"
    thumb_path.write_bytes(b"fake-png-bytes")

    uploader.set_default_thumbnail("vid123", thumb_path)

    _, call_kwargs = uploader.youtube.thumbnails.return_value.set.call_args
    assert call_kwargs["videoId"] == "vid123"
    fake_set.execute.assert_called_once()


def test_update_privacy_calls_update_with_status(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_update = MagicMock()
    uploader.youtube.videos.return_value.update.return_value = fake_update

    uploader.update_privacy("vid123", "public")

    _, call_kwargs = uploader.youtube.videos.return_value.update.call_args
    assert call_kwargs["part"] == "status"
    assert call_kwargs["body"] == {
        "id": "vid123",
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    fake_update.execute.assert_called_once()


def test_truncate_tags_by_char_budget():
    tags = ["a" * 60 for _ in range(10)]

    result = _truncate_tags(tags, max_chars=500)

    # 8 tags of 60 chars + 7 separators (1 per tag after the first) = 487 <= 500
    # A 9th tag would push it to 548 > 500, so it must be dropped.
    assert len(result) == 8
    assert result == tags[:8]


def test_truncate_tags_short_list_unchanged():
    tags = ["money psychology", "splurj"]

    result = _truncate_tags(tags, max_chars=500)

    assert result == tags


def _fake_token_cache(tmp_path, monkeypatch) -> None:
    """Point TOKEN_CACHE at a real (but content-irrelevant) file.

    has_valid_cached_token() only needs TOKEN_CACHE.exists() to be True;
    pickle.load is mocked separately per test so creds can be a MagicMock
    (MagicMocks aren't reliably picklable, so we never round-trip one through
    a real pickle file).
    """
    cache = tmp_path / "yt_token.pickle"
    cache.write_bytes(b"placeholder")
    monkeypatch.setattr("engine.youtube.TOKEN_CACHE", cache)


def test_has_valid_cached_token_false_when_no_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.youtube.TOKEN_CACHE", tmp_path / "missing.pickle")

    assert has_valid_cached_token() is False


def test_has_valid_cached_token_false_when_cache_unreadable(tmp_path, monkeypatch):
    _fake_token_cache(tmp_path, monkeypatch)

    with patch("engine.youtube.pickle.load", side_effect=pickle.UnpicklingError("bad data")):
        assert has_valid_cached_token() is False


def test_has_valid_cached_token_true_when_creds_already_valid(tmp_path, monkeypatch):
    _fake_token_cache(tmp_path, monkeypatch)
    creds = MagicMock(valid=True)

    with patch("engine.youtube.pickle.load", return_value=creds):
        assert has_valid_cached_token() is True


def test_has_valid_cached_token_refreshes_expired_creds(tmp_path, monkeypatch):
    _fake_token_cache(tmp_path, monkeypatch)
    creds = MagicMock(valid=False, expired=True, refresh_token="rt")

    with patch("engine.youtube.pickle.load", return_value=creds), \
         patch("engine.youtube.pickle.dump") as mock_dump, \
         patch("engine.youtube.Request"):
        assert has_valid_cached_token() is True

    creds.refresh.assert_called_once()
    mock_dump.assert_called_once()  # refreshed creds re-saved to the cache


def test_has_valid_cached_token_false_when_refresh_fails(tmp_path, monkeypatch):
    _fake_token_cache(tmp_path, monkeypatch)
    creds = MagicMock(valid=False, expired=True, refresh_token="rt")
    creds.refresh.side_effect = Exception("invalid_grant")

    with patch("engine.youtube.pickle.load", return_value=creds), \
         patch("engine.youtube.Request"):
        assert has_valid_cached_token() is False


def test_has_valid_cached_token_false_when_no_refresh_token(tmp_path, monkeypatch):
    _fake_token_cache(tmp_path, monkeypatch)
    creds = MagicMock(valid=False, expired=True, refresh_token=None)

    with patch("engine.youtube.pickle.load", return_value=creds):
        assert has_valid_cached_token() is False
