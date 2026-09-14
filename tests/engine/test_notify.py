import subprocess
from unittest.mock import MagicMock, patch

from engine.notify import notify_auth_failure, send_email, send_toast


def test_send_toast_invokes_powershell():
    with patch("engine.notify.subprocess.run") as mock_run:
        send_toast("Title", "Message")

    args, kwargs = mock_run.call_args
    assert args[0][0] == "powershell"
    assert "Title" in args[0][-1]
    assert "Message" in args[0][-1]


def test_send_toast_swallows_errors():
    with patch("engine.notify.subprocess.run", side_effect=subprocess.TimeoutExpired("powershell", 30)):
        send_toast("Title", "Message")  # must not raise


def test_send_email_skips_when_not_configured(monkeypatch):
    for key in ("NOTIFY_SMTP_HOST", "NOTIFY_SMTP_USER", "NOTIFY_SMTP_PASSWORD", "NOTIFY_EMAIL_TO"):
        monkeypatch.delenv(key, raising=False)

    with patch("engine.notify.smtplib.SMTP") as mock_smtp:
        send_email("Subject", "Body")

    mock_smtp.assert_not_called()


def test_send_email_sends_via_smtp_when_configured(monkeypatch):
    monkeypatch.setenv("NOTIFY_SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("NOTIFY_SMTP_PORT", "587")
    monkeypatch.setenv("NOTIFY_SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("NOTIFY_SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "me@gmail.com")

    mock_smtp_instance = MagicMock()
    mock_smtp_cm = MagicMock()
    mock_smtp_cm.__enter__.return_value = mock_smtp_instance

    with patch("engine.notify.smtplib.SMTP", return_value=mock_smtp_cm) as mock_smtp:
        send_email("Subject", "Body")

    mock_smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=20)
    mock_smtp_instance.starttls.assert_called_once()
    mock_smtp_instance.login.assert_called_once_with("me@gmail.com", "app-password")
    mock_smtp_instance.send_message.assert_called_once()


def test_send_email_swallows_smtp_errors(monkeypatch):
    monkeypatch.setenv("NOTIFY_SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("NOTIFY_SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("NOTIFY_SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "me@gmail.com")

    with patch("engine.notify.smtplib.SMTP", side_effect=OSError("connection refused")):
        send_email("Subject", "Body")  # must not raise


def test_notify_auth_failure_calls_both_channels():
    with patch("engine.notify.send_toast") as mock_toast, \
         patch("engine.notify.send_email") as mock_email:
        notify_auth_failure("Token is dead.")

    mock_toast.assert_called_once()
    mock_email.assert_called_once()
    assert mock_toast.call_args[0][1] == mock_email.call_args[0][1]  # same message to both
