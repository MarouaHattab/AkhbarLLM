from typing import List, Dict
import json

from src.entites.TranslatedStory import TranslatedStory

LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "ar": "Arabic",
    "arabic": "Arabic",
}


def normalize_language(language: str, field_name: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"{field_name} is empty or None")

    normalized = language.strip()
    return LANGUAGE_ALIASES.get(normalized.casefold(), normalized)


def build_translation_prompt(
    story: str,
    target_language: str = "ar",
    source_language: str = "english",
) -> List[Dict]:
    if not isinstance(story, str) or not story.strip():
        raise ValueError("Story is empty or None")

    source_language = normalize_language(source_language, "Source language")
    target_language = normalize_language(target_language, "Target language")
   
    messages =[
    {
        "role": "system",
        "content": "\n".join([
            "You are a professional translator.",
            "You will be provided by an {} text.",
            "You have to translate the text into the `Targeted Language`.",
            "Follow the provided Scheme to generate a JSON",
            "Do not generate any introduction or conclusion."
        ]).format(source_language)
    },
    {
        "role": "user",
        "content":  "\n".join([
            "## Story:",
            story.strip(),
            "",

            "## Pydantic Details:",
            json.dumps( TranslatedStory.model_json_schema(), ensure_ascii=False ),
            "",

            "## Targeted Language:",
            target_language,
            "",

            "## Translated Story:",
            "```json"

        ])
    }
]
    return messages
