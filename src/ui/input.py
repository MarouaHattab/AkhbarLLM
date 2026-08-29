from pathlib import Path


SUPPORTED_STORY_SUFFIXES = {".txt", ".md"}


def decode_uploaded_story(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_STORY_SUFFIXES:
        raise ValueError("Upload a UTF-8 TXT or Markdown file.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    try:
        story = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded file must use UTF-8 encoding.") from exc
    if not story:
        raise ValueError("The uploaded file contains no story text.")
    return story


def resolve_story_input(
    pasted_story: str,
    *,
    uploaded_name: str | None,
    uploaded_bytes: bytes | None,
) -> str:
    pasted = pasted_story.strip()
    if pasted:
        return pasted
    if uploaded_name is not None and uploaded_bytes is not None:
        return decode_uploaded_story(uploaded_name, uploaded_bytes)
    raise ValueError("Paste a story or upload a UTF-8 TXT/Markdown file.")
