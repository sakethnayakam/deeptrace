"""Create and validate research plans using Gemini."""

import json

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.services.gemini import GeminiService, GeminiServiceError


MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 6
MAX_SUBQUESTION_LENGTH = 500


class ResearchPlannerError(Exception):
    """Raised when Gemini cannot provide a valid research plan."""


class GeneratedResearchPlan(BaseModel):
    """The JSON shape expected from Gemini."""

    subquestions: list[str] = Field(
        min_length=MIN_SUBQUESTIONS,
        max_length=MAX_SUBQUESTIONS,
    )

    @field_validator("subquestions")
    @classmethod
    def subquestions_must_be_useful(cls, subquestions: list[str]) -> list[str]:
        cleaned_subquestions: list[str] = []
        seen_subquestions: set[str] = set()

        for subquestion in subquestions:
            cleaned_subquestion = subquestion.strip()
            if not cleaned_subquestion:
                raise ValueError("Subquestions must not be blank.")
            if len(cleaned_subquestion) > MAX_SUBQUESTION_LENGTH:
                raise ValueError("Subquestions are too long.")

            normalized_subquestion = cleaned_subquestion.casefold()
            if normalized_subquestion in seen_subquestions:
                raise ValueError("Subquestions must be distinct.")

            seen_subquestions.add(normalized_subquestion)
            cleaned_subquestions.append(cleaned_subquestion)

        return cleaned_subquestions


class ResearchPlanner:
    """Turn one research question into a structured investigation plan."""

    def __init__(self, gemini_service: GeminiService) -> None:
        self.gemini_service = gemini_service

    def plan(self, question: str) -> list[str]:
        """Return validated research subquestions for the given question."""
        try:
            generated_text = self.gemini_service.generate_json(
                self._build_prompt(question)
            )
        except GeminiServiceError as error:
            raise ResearchPlannerError("Gemini could not create a plan") from error

        try:
            generated_plan = GeneratedResearchPlan.model_validate_json(generated_text)
        except (ValidationError, ValueError) as error:
            raise ResearchPlannerError("Gemini returned an invalid plan") from error

        return generated_plan.subquestions

    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"""You are a research planner.

Break the research question below into {MIN_SUBQUESTIONS} to {MAX_SUBQUESTIONS}
distinct, researchable subquestions. Cover useful dimensions such as definitions,
evidence, comparisons, and limitations when relevant.

Return JSON only in this exact shape:
{{"subquestions": ["question one", "question two", "question three"]}}

Research question: {question}
"""
