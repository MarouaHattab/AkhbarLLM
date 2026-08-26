from pathlib import Path


def load_story(path: str | Path) -> str:
    """Load a UTF-8 story from plain text or the legacy notebook format."""
    story_path = Path(path)
    text = story_path.read_text(encoding="utf-8").strip()

    prefix = 'story = """'
    if text.startswith(prefix) and text.endswith('"""'):
        text = text[len(prefix) : -3].strip()

    if not text:
        raise ValueError(f"Story file is empty: {story_path}")

    return text
