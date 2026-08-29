import json

from src.models.translation import TranslatedStory
from src.templates import TRANSLATION_SYSTEM_PROMPT_TEMPLATE


LANGUAGE_ALIASES = {
    "ar": "Arabic",
    "arabic": "Arabic",
    "en": "English",
    "english": "English",
    "fr": "French",
    "french": "French",
}


def normalize_language(language: str, field_name: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    normalized = language.strip()
    return LANGUAGE_ALIASES.get(normalized.casefold(), normalized)


def build_translation_messages(
    story: str,
    target_language: str = "English",
    source_language: str = "Arabic",
) -> list[dict[str, str]]:
    if not isinstance(story, str) or not story.strip():
        raise ValueError("Story must be a non-empty string.")

    normalized_source = normalize_language(source_language, "Source language")
    normalized_target = normalize_language(target_language, "Target language")
    system_prompt = TRANSLATION_SYSTEM_PROMPT_TEMPLATE.format(
        source_language=normalized_source
    )

    user_prompt = "\n".join(
        [
            "## Story:",
            story.strip(),
            "",
            "## Output Schema:",
            json.dumps(
                TranslatedStory.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "## Target Language:",
            normalized_target,
            "",
            "## Translated Story:",
        ]
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
