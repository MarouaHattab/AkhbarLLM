import json
from typing import List, Dict
from src.entites.ExtractedNews import NewsDetails
def build_extraction_prompt(story: str) -> List[Dict]:
    if not isinstance(story, str) or not story.strip():
        raise ValueError("Story is empty or None")

    messages =[
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data paraser.",
                "You will be provided by an Arabic text associated with a Pydantic scheme.",
                "Generate the ouptut in the same story language.",
                "You have to extract JSON details from text according the Pydantic details.",
                "Extract details as mentioned in text.",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
           "role": "user",
            "content":  "\n".join([
            "## Story:",
            story.strip(),
            "",
                "## Pydantic Details:",
                json.dumps(
                    NewsDetails.model_json_schema(), ensure_ascii=False
                ),
                "",

                "## Story Details:",
                "```json"
            ])
        }
    ]
    return messages
