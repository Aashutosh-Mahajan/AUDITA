"""
Centralized Anthropic API wrapper with structured output support.

All LLM calls flow through this module — cleaning plan, insight planning,
and grounding checks. Retry/backoff logic is centralized here so nodes
don't duplicate API boilerplate.
"""

import json
import os
import time
from typing import Any, TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Lazy-initialise and return the Anthropic client singleton."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Pydantic model → Anthropic tool schema conversion
# ---------------------------------------------------------------------------


def _pydantic_to_tool_schema(
    model_class: type[BaseModel],
    tool_name: str,
    description: str,
) -> dict[str, Any]:
    """Convert a Pydantic model class to an Anthropic tool definition."""
    schema = model_class.model_json_schema()
    # Remove Pydantic-specific keys that Anthropic doesn't need
    schema.pop("title", None)

    return {
        "name": tool_name,
        "description": description,
        "input_schema": schema,
    }


def _pydantic_list_to_tool_schema(
    model_class: type[BaseModel],
    tool_name: str,
    description: str,
) -> dict[str, Any]:
    """Create a tool schema that expects a list of Pydantic model instances."""
    item_schema = model_class.model_json_schema()
    # Inline $defs if present (Anthropic tools don't support $ref)
    defs = item_schema.pop("$defs", {})

    return {
        "name": tool_name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": item_schema,
                    "description": f"List of {model_class.__name__} objects",
                }
            },
            "required": ["items"],
            # Embed $defs at the top level for sub-schema resolution
            **({"$defs": defs} if defs else {}),
        },
    }


# ---------------------------------------------------------------------------
# Core structured-output call
# ---------------------------------------------------------------------------


def call_structured(
    system_prompt: str,
    user_prompt: str,
    tool_schema: dict[str, Any],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Make an Anthropic API call expecting structured tool-use output.

    Returns the parsed tool input dict (validated JSON, not yet Pydantic).
    Retries on transient API errors with exponential backoff.
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": tool_schema["name"]},
            )

            # Extract tool use block
            for block in response.content:
                if block.type == "tool_use":
                    return block.input

            raise ValueError("No tool_use block in LLM response")

        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Anthropic API failed after {MAX_RETRIES} retries: {e}"
                ) from e

    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# High-level helpers used by nodes
# ---------------------------------------------------------------------------


def request_cleaning_plan(
    quality_audit: dict[str, Any],
    column_names: list[str],
    cleaning_action_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """LLM call #1 — request a list of cleaning actions.

    Returns a list of raw dicts (each matching CleaningAction shape).
    Validation against real columns is done by the calling node, not here.
    """
    system_prompt = (
        "You are a data-cleaning assistant. You will be given a quality audit "
        "of a dataset. Propose cleaning actions using ONLY the following action types:\n"
        "  impute_mean, impute_median, impute_mode, drop_rows, drop_column, "
        "standardize_categories, parse_dates, cap_outliers, no_action\n\n"
        "Rules:\n"
        "- Only propose actions for columns that are PRESENT in the audit.\n"
        "- Do NOT invent column names.\n"
        "- Each action must include a rationale explaining why it's appropriate.\n"
        "- If a column needs no cleaning, use no_action.\n"
        f"- Available columns: {column_names}"
    )

    user_prompt = (
        "Here is the quality audit for the dataset:\n\n"
        f"```json\n{json.dumps(quality_audit, indent=2)}\n```\n\n"
        "Propose a list of cleaning actions."
    )

    result = call_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_schema=cleaning_action_schema,
    )

    return result.get("items", [result])


def request_viz_intents(
    clean_profile: dict[str, Any],
    column_names: list[str],
    viz_intent_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """LLM call #2 — request a ranked list of visualization intents.

    Returns a list of raw dicts (each matching VizIntent shape).
    Validation against columns and dtypes is done by the calling node.
    """
    system_prompt = (
        "You are a data-visualization assistant. You will be given a profile "
        "of a cleaned dataset. Propose visualizations using ONLY these chart types:\n"
        "  histogram, box, bar, line, scatter, heatmap, pie\n\n"
        "Rules:\n"
        "- Only reference columns that EXIST in the profile.\n"
        "- Assign a priority_score from 0 to 5 (higher = more insightful).\n"
        "- Assign a category: distribution, relationship, trend, or categorical.\n"
        "- Include a rationale for each visualization.\n"
        "- Propose diverse chart types — don't repeat the same type unless it "
        "  genuinely adds different insight.\n"
        f"- Available columns: {column_names}"
    )

    user_prompt = (
        "Here is the profile of the cleaned dataset:\n\n"
        f"```json\n{json.dumps(clean_profile, indent=2)}\n```\n\n"
        "Propose a ranked list of visualizations."
    )

    result = call_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_schema=viz_intent_schema,
    )

    return result.get("items", [result])


def request_grounding_check(
    chart_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """LLM grounding check — batch verify that charts support their rationale.

    Accepts a list of chart summaries; returns a list of
    ``{"grounded": bool, "notes": str}`` in the same order.

    Batches into a single call when len > 1 (cost control per §8).
    """
    system_prompt = (
        "You are a chart verification assistant. For each chart summary, "
        "determine whether the chart faithfully supports its stated rationale. "
        "Return a list of verdicts in the SAME ORDER as the input."
    )

    user_prompt = (
        f"Verify these charts:\n\n```json\n{json.dumps(chart_summaries, indent=2)}\n```"
    )

    tool_schema = {
        "name": "grounding_verdicts",
        "description": "List of grounding verdicts, one per chart",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grounded": {"type": "boolean"},
                            "notes": {"type": "string"},
                        },
                        "required": ["grounded", "notes"],
                    },
                }
            },
            "required": ["verdicts"],
        },
    }

    result = call_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_schema=tool_schema,
    )

    return result.get("verdicts", [])
