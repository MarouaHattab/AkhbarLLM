from pydantic import BaseModel, ConfigDict, Field


class TranslatedStory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translated_title: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Suggested translated title of the news story.",
    )
    translated_content: str = Field(
        ...,
        min_length=5,
        description="Translated content of the news story.",
    )
