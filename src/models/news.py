from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


StoryCategory = Literal[
    "politics",
    "sports",
    "art",
    "technology",
    "economy",
    "health",
    "entertainment",
    "science",
    "not_specified",
]

EntityType = Literal[
    "person-male",
    "person-female",
    "location",
    "organization",
    "event",
    "time",
    "quantity",
    "money",
    "product",
    "law",
    "disease",
    "artifact",
    "not_specified",
]


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_value: str = Field(
        ...,
        description="The actual name or value of the entity.",
    )
    entity_type: EntityType = Field(
        ...,
        description="The type of recognized entity.",
    )


class NewsDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_title: str = Field(
        ...,
        min_length=5,
        max_length=100,
        description="A fully informative and SEO-optimized title of the story.",
    )
    story_keywords: list[str] = Field(
        ...,
        min_length=1,
        description="Relevant keywords associated with the story.",
    )
    story_summary: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Summarized key points about the story (1-5 points).",
    )
    story_category: StoryCategory = Field(
        ...,
        description="Category of the news story.",
    )
    story_entities: list[Entity] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of identified entities in the story.",
    )
