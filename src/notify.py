import shutil
import subprocess


def notify(title, text):
    text = str(text).replace('"', "'")[:200]
    title = str(title).replace('"', "'")[:60]
    try:
        if shutil.which("osascript"):
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application "Terminal" to display notification '
                    f'"{text}" with title "{title}" sound name "Glass"',
                ],
                capture_output=True,
                timeout=5,
            )
        elif shutil.which("notify-send"):
            subprocess.run(
                ["notify-send", title, text], capture_output=True, timeout=5
            )
    except Exception:
        pass
