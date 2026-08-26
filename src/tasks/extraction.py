import json

from src.models import NewsDetails
from src.templates import EXTRACTION_SYSTEM_PROMPT


def build_extraction_messages(story: str) -> list[dict[str, str]]:
    if not isinstance(story, str) or not story.strip():
        raise ValueError("Story must be a non-empty string.")

    user_prompt = "\n".join(
        [
            "## Story:",
            story.strip(),
            "",
            "## Output Schema:",
            json.dumps(
                NewsDetails.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "## Story Details:",
        ]
    )

    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
