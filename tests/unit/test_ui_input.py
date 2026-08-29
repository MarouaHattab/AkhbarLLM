import pytest

from src.ui.input import decode_uploaded_story, resolve_story_input


def test_decode_uploaded_story_accepts_utf8_bom() -> None:
    data = b"\xef\xbb\xbf\xd8\xae\xd8\xa8\xd8\xb1"
    assert decode_uploaded_story("story.txt", data) == "خبر"


@pytest.mark.parametrize("filename", ["story.txt", "story.md", "STORY.TXT"])
def test_decode_uploaded_story_accepts_supported_extensions(filename: str) -> None:
    assert decode_uploaded_story(filename, "قصة".encode("utf-8")) == "قصة"


def test_pasted_story_takes_precedence() -> None:
    assert resolve_story_input(
        " pasted story ",
        uploaded_name="story.txt",
        uploaded_bytes=b"uploaded story",
    ) == "pasted story"


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("story.pdf", b"content", "TXT or Markdown"),
        ("story.txt", b"", "empty"),
        ("story.txt", b"\xff", "UTF-8"),
    ],
)
def test_decode_uploaded_story_reports_precise_input_errors(
    filename: str,
    data: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_uploaded_story(filename, data)


def test_resolve_story_input_rejects_empty_submission() -> None:
    with pytest.raises(ValueError, match="Paste a story or upload"):
        resolve_story_input("", uploaded_name=None, uploaded_bytes=None)
