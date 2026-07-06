"""
YouTube Data API v3 upload module for Splurj.

Auth flow: OAuth2 via a client_secret JSON file from Google Cloud Console.
Token is cached at ~/.splurj/yt_token.pickle so the browser prompt only
fires once (or when the refresh token expires).

This module is format-agnostic: whether a video is the long-form upload or
an auto-cut Short, and what URL framing/hashtags to log, is the caller's
decision (see splurj_engine.py) — not this module's.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube"]
TOKEN_CACHE = Path.home() / ".splurj" / "yt_token.pickle"
DEFAULT_CATEGORY = "27"  # Education


class YouTubeUploader:
    def __init__(self, client_secret_file: str, privacy_status: str = "private"):
        self.client_secret_file = client_secret_file
        self.privacy_status = privacy_status
        self.youtube = self._authenticate()

    def _authenticate(self):
        creds: Optional[Credentials] = None
        TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)

        if TOKEN_CACHE.exists():
            with open(TOKEN_CACHE, "rb") as fh:
                creds = pickle.load(fh)

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("YouTube token refreshed.")
            except Exception as exc:
                logger.warning("Token refresh failed (%s); re-authenticating.", exc)
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True)
            logger.info("YouTube OAuth complete.")

        with open(TOKEN_CACHE, "wb") as fh:
            pickle.dump(creds, fh)

        return build("youtube", "v3", credentials=creds)

    def upload(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = DEFAULT_CATEGORY,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags[:500],
                "categoryId": category_id,
                "defaultLanguage": "en",
            },
            "status": {
                "privacyStatus": self.privacy_status,
                "selfDeclaredMadeForKids": False,
                "madeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(video_path), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024,
        )

        logger.info("Starting upload: '%s' [%s]", title, self.privacy_status)
        request = self.youtube.videos().insert(
            part=",".join(body.keys()), body=body, media_body=media,
        )

        response = self._resumable_upload(request)
        logger.info("Upload complete -> video id %s", response.get("id", "unknown"))
        return response

    def set_default_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        media = MediaFileUpload(str(thumbnail_path), mimetype="image/png")
        self.youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        logger.info("Thumbnail set for video %s -> %s", video_id, thumbnail_path.name)

    def update_privacy(self, video_id: str, privacy_status: str) -> None:
        self.youtube.videos().update(
            part="status",
            body={"id": video_id, "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False}},
        ).execute()
        logger.info("Video %s -> %s", video_id, privacy_status)

    def _resumable_upload(self, request, max_retries: int = 6) -> Dict[str, Any]:
        response = None
        retry = 0

        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    logger.info("Upload progress: %d%%", int(status.progress() * 100))
            except HttpError as exc:
                if exc.resp.status in {500, 502, 503, 504} and retry < max_retries:
                    retry += 1
                    wait = 2 ** retry
                    logger.warning("HTTP %s — retry %d/%d in %ds", exc.resp.status, retry, max_retries, wait)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Upload failed after {retry} retries: {exc}") from exc
            except Exception as exc:
                if retry < max_retries:
                    retry += 1
                    wait = 2 ** retry
                    logger.warning("Upload error — retry %d/%d: %s", retry, max_retries, exc)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Upload aborted: {exc}") from exc

        return response
