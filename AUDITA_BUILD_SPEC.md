# AUDITA — Build Specification for Coding Agent

**Project:** AUDITA — Auditable, Self-Verifying Data Cleaning & Visualization Agent
**Stack:** Python, LangGraph, Anthropic API, pandas, Plotly, Streamlit
**Purpose of this document:** Implementation instructions for a coding agent. Read fully before writing any code. This is not a PRD — it is a build spec. Follow the architecture decisions exactly; do not substitute your own patterns (e.g. do not let the LLM generate plotting/cleaning code directly — see Core Principle below).

---

## 0. Core Principle (do not violate)

**The LLM proposes. Code executes. Code verifies.**

The LLM is only ever used for two structured-output decisions:
1. Proposing cleaning actions (from a fixed enum, targeting real column names).
2. Proposing visualization intents (from a fixed enum, targeting real column names).

The LLM **never** writes or executes pandas code, plotting code, or arbitrary Python. All execution is deterministic, hand-written functions dispatched from a fixed action/chart-type registry. All LLM output must be schema-validated (Pydantic) against the real DataFrame's actual columns/dtypes before it is allowed to execute. Invalid proposals are rejected and re-requested, not silently coerced.

If you find yourself about to have the LLM generate a plotting/cleaning code snippet — stop. That is the wrong pattern for this project.

---

## 1. Repository Structure

Build this exact structure:

```
audita/
├── app.py                          # Streamlit entrypoint
├── requirements.txt
├── .env.example                    # ANTHROPIC_API_KEY=
├── graph/
│   ├── __init__.py
│   ├── state.py                    # PipelineState TypedDict
│   ├── build_graph.py              # graph wiring, Send() fan-out, conditional edges
│   └── nodes/
│       ├── __init__.py
│       ├── ingest.py
│       ├── quality_audit.py
│       ├── cleaning_plan.py        # LLM call
│       ├── cleaning_exec.py
│       ├── profiling.py
│       ├── insight_planning.py     # LLM call
│       ├── chart_builder.py        # fan-out target
│       ├── self_check.py           # code check + LLM grounding call
│       └── assemble.py
├── core/
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic models
│   ├── cleaning_registry.py        # action_type -> pandas function dispatch
│   ├── chart_registry.py           # chart_type -> Plotly render function dispatch
│   ├── audit_log.py                # append-only audit trail helper
│   └── llm_client.py               # Anthropic API wrapper, structured output helper
├── ui/
│   ├── __init__.py
│   ├── components.py                # Streamlit rendering helpers (cards, badges, tables)
│   └── sections.py                  # dashboard section renderers
└── tests/
    ├── test_cleaning_registry.py
    ├── test_chart_registry.py
    └── test_schemas.py
```

---

## 2. Pydantic Schemas (`core/schemas.py`)

Implement these exactly; other modules depend on these shapes.

```python
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field

class CleaningActionType(str, Enum):
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    DROP_ROWS = "drop_rows"
    DROP_COLUMN = "drop_column"
    STANDARDIZE_CATEGORIES = "standardize_categories"
    PARSE_DATES = "parse_dates"
    CAP_OUTLIERS = "cap_outliers"
    NO_ACTION = "no_action"

class CleaningAction(BaseModel):
    column: str
    action_type: CleaningActionType
    rationale: str
    params: dict[str, Any] = Field(default_factory=dict)  # e.g. {"cap_percentile": 0.99}

class CleaningDiffEntry(BaseModel):
    column: str
    action_type: CleaningActionType
    rows_affected: int
    before_stat: dict[str, Any]
    after_stat: dict[str, Any]

class ChartType(str, Enum):
    HISTOGRAM = "histogram"
    BOX = "box"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"

class VizIntent(BaseModel):
    chart_type: ChartType
    columns: list[str]
    rationale: str
    priority_score: float = Field(ge=0, le=5)
    category: str  # "distribution" | "relationship" | "trend" | "categorical"

class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FLAGGED = "flagged"
    RETRYING = "retrying"
    FAILED = "failed"

class ChartResult(BaseModel):
    intent: VizIntent
    figure_json: Optional[str] = None   # Plotly figure.to_json()
    verification_status: VerificationStatus
    verification_notes: str = ""
    retry_count: int = 0
    error: Optional[str] = None

class AuditLogEntry(BaseModel):
    timestamp: str
    stage: str
    actor: str  # "llm" | "code"
    action: str
    detail: dict[str, Any]
```

Every LLM structured-output call must return objects validated against `CleaningAction` / `VizIntent` — use Anthropic's tool-use / structured output mode, not free-text parsing.

---

## 3. LangGraph State (`graph/state.py`)

```python
from typing import TypedDict, Annotated
import operator
from core.schemas import CleaningAction, CleaningDiffEntry, VizIntent, ChartResult, AuditLogEntry

class PipelineState(TypedDict):
    csv_path: str
    raw_profile: dict
    quality_audit: dict
    cleaning_plan: list[CleaningAction]
    cleaned_csv_path: str
    cleaning_diff: list[CleaningDiffEntry]
    clean_profile: dict
    proposed_visualizations: list[VizIntent]
    completed_charts: Annotated[list[ChartResult], operator.add]  # fan-in accumulator
    audit_log: Annotated[list[AuditLogEntry], operator.add]
    human_approved_cleaning: bool  # gate flag, see Section 6
```

`completed_charts` and `audit_log` use `operator.add` because they are written to concurrently by parallel `Send()`-dispatched nodes — this is required for LangGraph's fan-in to work correctly. Do not use a plain `list` type for these two fields.

---

## 4. Pipeline Stages — Implementation Notes

Build nodes in this order; each depends on the previous being correct.

### 4.1 `ingest.py`
- Use `pandas.read_csv` with `csv.Sniffer` for delimiter detection; fall back to `chardet` for encoding if UTF-8 decode fails.
- Output: DataFrame (kept in memory / temp path) + `raw_profile` = `{n_rows, n_cols, file_size_bytes, columns: [{name, dtype}]}`.

### 4.2 `quality_audit.py` (pure code, no LLM)
Per column compute:
- `missing_pct`, `dtype`, `n_unique`
- numeric columns: `mean, std, min, max, iqr_outlier_count`
- string/categorical columns: top value counts, fuzzy near-duplicate labels (use `rapidfuzz` or `difflib.SequenceMatcher`, threshold ~0.85)
- columns that look like dates but are typed as string: date-parse success rate via `pd.to_datetime(..., errors="coerce")`

Output a JSON-serializable `quality_audit` dict. **This dict — not the raw CSV — is what gets shown to the LLM in the next stage.** Never put the full DataFrame in an LLM prompt.

### 4.3 `cleaning_plan.py` (LLM call #1)
- Prompt: system message describes the fixed `CleaningActionType` enum and the rule "only propose actions for columns present in the audit; do not invent columns."
- Use Anthropic tool-use (function calling) with a tool schema mirroring `CleaningAction`, requesting a list.
- After the LLM responds: **validate every returned `column` against `raw_profile.columns`**. Drop/reject any action referencing a nonexistent column, log the rejection to `audit_log`, and either re-prompt once or skip that column with a `NO_ACTION` fallback — do not crash the pipeline on one bad proposal.

### 4.4 `cleaning_exec.py` (pure code — `core/cleaning_registry.py`)
Implement a dispatch dict:
```python
CLEANING_REGISTRY: dict[CleaningActionType, Callable[[pd.DataFrame, CleaningAction], pd.DataFrame]] = {
    CleaningActionType.IMPUTE_MEAN: _impute_mean,
    CleaningActionType.IMPUTE_MEDIAN: _impute_median,
    ...
}
```
Each function: takes `(df, action)`, returns `df` (or a copy), and the calling node must snapshot before/after stats for that column to build a `CleaningDiffEntry`. Run actions sequentially, accumulate diffs, append one `AuditLogEntry` per action.

### 4.5 `profiling.py`
Re-run the exact same audit function from 4.2 on the cleaned DataFrame → `clean_profile`. Reusing the same function (not a near-duplicate) is required so before/after are directly comparable in the UI.

### 4.6 `insight_planning.py` (LLM call #2)
- Prompt: system message describes the fixed `ChartType` enum, the `clean_profile`, and asks for a ranked list of `VizIntent`s with `priority_score` (0–5) and `category`.
- Validate every intent's `columns` against `clean_profile` columns and against dtype compatibility rules (e.g. scatter needs 2 numeric columns; heatmap needs ≥2 numeric columns; bar/pie need a categorical + optionally a numeric) — implement this as a deterministic compatibility table in `core/chart_registry.py`, not an LLM judgment.
- After validation, apply a deterministic filter: drop intents with `priority_score < 2.5`, and deduplicate intents that target the same column set + category (keep highest score).

### 4.7 `chart_builder.py` (fan-out via `Send()`)
- In `build_graph.py`, use a conditional edge function that returns a list of `Send("build_chart", {...intent, "clean_csv_path": ...})` — one per approved `VizIntent` — so LangGraph runs them concurrently.
- Each invocation: look up `chart_type` in `core/chart_registry.py`'s `CHART_REGISTRY` dispatch dict, call the matching `render_*(df, intent) -> plotly.graph_objects.Figure` function. Wrap in try/except; on failure return a `ChartResult` with `verification_status=FAILED` and the error message rather than raising.
- Registry functions are hand-written, tested Plotly calls — one function per `ChartType`. No LLM involvement in this node at all.

### 4.8 `self_check.py` (code check + LLM grounding call)
Two layers, both must run:
1. **Code-level recomputation**: independently recompute the statistic the chart claims to show directly from the cleaned DataFrame (e.g. for a bar chart of group means, recompute `df.groupby(col)[val].mean()` separately and diff against the values embedded in the Plotly figure JSON). Also check for a truncated/misleading y-axis (does the axis range start meaningfully above 0 for a bar/column chart without a stated reason).
2. **LLM grounding check**: send the chart's `intent.rationale`, `chart_type`, `columns`, and a compact textual description of the rendered figure (title, axis labels, a few data points) — not the full dataset — and ask the LLM to confirm the chart supports the stated rationale. Use structured output: `{"grounded": bool, "notes": str}`.
- If either check fails: increment `retry_count`, attach the failure reason, and route back to `chart_builder` for that single intent (bounded — max 2 retries, configurable constant `MAX_CHART_RETRIES`). If still failing after max retries, set `verification_status=FLAGGED` (not FAILED — a flagged chart still renders, but with a visible warning, so the user isn't silently missing content).

### 4.9 `assemble.py`
Pure code. Reads final `PipelineState` and produces whatever structure `ui/sections.py` needs (grouped charts by `category`, cleaning diff table, audit log table). No LLM calls here.

---

## 5. Graph Wiring (`graph/build_graph.py`)

Linear edges: `ingest → quality_audit → cleaning_plan → [human gate] → cleaning_exec → profiling → insight_planning → [fan-out to chart_builder] → self_check → [conditional: retry or continue] → assemble`.

Key LangGraph mechanics to get right:
- Fan-out: use `Send()` from the conditional edge after `insight_planning`, one dispatch per approved `VizIntent`.
- Fan-in: `completed_charts` uses `operator.add` in state (see Section 3) so parallel branches merge automatically into one list before `self_check`/`assemble` run.
- Retry loop: `self_check` should be able to route back to a single-chart rebuild, not the whole graph — scope the conditional edge to the specific failed `ChartResult`, not a full re-invoke.
- Compile the graph with a checkpointer if you want mid-pipeline resumability (e.g. resuming after the human cleaning-approval gate) — recommended given Section 6 below.

---

## 6. Human-in-the-loop Gate (cleaning approval)

After `cleaning_plan`, do not auto-execute. Return the proposed `cleaning_plan` to the Streamlit UI, let the user check/uncheck individual `CleaningAction`s, then resume the graph into `cleaning_exec` with only the approved subset. Implement this as a LangGraph interrupt (`interrupt_before=["cleaning_exec"]` at compile time) so the graph pauses cleanly rather than requiring you to hand-roll state persistence.

---

## 7. Streamlit App (`app.py`)

- File upload widget → on upload, if no cached result exists for this file hash in `st.session_state`, invoke the graph (`graph.stream(...)` for live progress, or `graph.invoke(...)` for a single blocking call — prefer `stream()` so the UI can show stage-by-stage progress: "Auditing data quality... Proposing cleaning plan... Building 7 charts...").
- Cache the full final state in `st.session_state[f"result_{file_hash}"]`.
- All subsequent widget interactions (chart filters, expanding audit log rows) must render from cached state only — **never re-invoke the graph on a normal Streamlit rerun.** Only explicit buttons ("Re-run cleaning with edited plan", "Re-run insight planning") should re-invoke, and only from the relevant node onward.
- Dashboard sections (via `ui/sections.py`): (1) data quality summary before/after, (2) cleaning diff table with rationale, (3) charts grouped by `category`, each with a visible badge for `verification_status` (green = verified, yellow = flagged with notes shown on hover/expand), (4) collapsible full audit log.

---

## 8. LLM Client (`core/llm_client.py`)

- Single wrapper around the Anthropic API using tool-use / structured output for all two structured calls (cleaning plan, insight planning) plus the grounding check in `self_check.py`.
- Centralize model name, `max_tokens`, and retry/backoff logic here — don't duplicate API-call boilerplate in each node.
- Batch the grounding check across all charts in one call when there are more than ~5 charts, to control cost (pass a list of chart summaries, expect a list of `{"grounded": bool, "notes": str}` back in the same order) — implement this as the default path, not an optimization to defer.

---

## 9. Testing Expectations

- `tests/test_cleaning_registry.py`: each `CleaningActionType` handler tested against a small synthetic DataFrame with known missing/outlier values; assert exact expected output.
- `tests/test_chart_registry.py`: each `ChartType` render function tested for (a) successful figure generation on compatible dtypes, (b) raising/returning a clean error on incompatible dtypes (e.g. scatter on two string columns).
- `tests/test_schemas.py`: Pydantic validation rejects out-of-enum values and out-of-range `priority_score`.
- Do not skip the registry tests — they're what makes the "code executes deterministically" guarantee actually true rather than assumed.

---

## 10. Explicit Non-Goals for This Build

Do not implement in v1 (see PRD Section 3.2 / 10 for rationale — these are intentionally deferred, not forgotten):
- Arbitrary LLM-generated plotting/cleaning code or a sandboxed code-exec path.
- Multi-file upload / join inference.
- Live database connections or streaming data.
- Per-chart natural-language follow-up / targeted re-analysis (Future Scope 10.1).

---

## 11. Build Order (suggested)

1. `core/schemas.py` — lock these first, everything depends on them.
2. `core/cleaning_registry.py` + `core/chart_registry.py` with tests — pure functions, no LLM, fastest to get right and verify.
3. `graph/state.py`.
4. Nodes in the order listed in Section 4, wiring each into `build_graph.py` incrementally and testing with a small sample CSV before adding the next node.
5. `core/llm_client.py` + wire into `cleaning_plan.py`, `insight_planning.py`, `self_check.py`.
6. Human-in-the-loop interrupt (Section 6).
7. `app.py` + `ui/` — build against a fully working graph, not in parallel with it.
