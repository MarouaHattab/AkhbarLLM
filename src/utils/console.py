import os
import sys
from typing import Any, TextIO


def _configure_windows_console_utf8() -> None:
    if sys.platform != "win32":
        return

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        return


def configure_utf8_output(*outputs: TextIO) -> None:
    """Ensure Python and the Windows console use UTF-8 for text output."""
    _configure_windows_console_utf8()

    for output in outputs:
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
