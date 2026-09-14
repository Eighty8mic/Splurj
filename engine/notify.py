"""Best-effort alerts for unattended nightly-run failures.

A notification failing to send is never a reason to fail the caller -- both
channels below swallow their own errors and just log a warning.
"""
import logging
import os
import smtplib
import subprocess
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def send_toast(title: str, message: str) -> None:
    """Show a Windows desktop balloon notification via PowerShell."""
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$ni = New-Object System.Windows.Forms.NotifyIcon; "
        "$ni.Icon = [System.Drawing.SystemIcons]::Warning; "
        "$ni.Visible = $true; "
        f"$ni.ShowBalloonTip(15000, '{title}', '{message}', "
        "[System.Windows.Forms.ToolTipIcon]::Warning); "
        "Start-Sleep -Seconds 16; "
        "$ni.Dispose()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True, capture_output=True, timeout=30,
        )
    except Exception as exc:
        logger.warning("Toast notification failed: %s", exc)


def send_email(subject: str, body: str) -> None:
    """Send an alert email via SMTP, configured through NOTIFY_SMTP_* env vars."""
    host = os.getenv("NOTIFY_SMTP_HOST")
    user = os.getenv("NOTIFY_SMTP_USER")
    password = os.getenv("NOTIFY_SMTP_PASSWORD")
    to_addr = os.getenv("NOTIFY_EMAIL_TO", user)
    port = int(os.getenv("NOTIFY_SMTP_PORT", "587"))

    if not (host and user and password and to_addr):
        logger.warning("Email notification skipped -- NOTIFY_SMTP_* not configured in .env")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
    except Exception as exc:
        logger.warning("Email notification failed: %s", exc)


def notify_auth_failure(detail: str) -> None:
    title = "Splurj: YouTube auth expired"
    message = f"{detail} Re-run the OAuth login command to fix it."
    send_toast(title, message)
    send_email(title, message)
