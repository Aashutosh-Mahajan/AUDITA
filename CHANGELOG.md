# Changelog

All notable changes to **AUDITA** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-09

### Added
- **Core Architecture**:
  - Pydantic models for all pipeline data schemas (`CleaningAction`, `VizIntent`, `ChartResult`, `AuditLogEntry`, `CleaningDiffEntry`).
  - Fixed enums (`CleaningActionType`, `ChartType`, `VerificationStatus`) ensuring strictly deterministic dispatch.
- **Deterministic Registries**:
  - `CLEANING_REGISTRY`: 9 hand-written pandas cleaning handlers (`impute_mean`, `impute_median`, `impute_mode`, `drop_rows`, `drop_column`, `standardize_categories`, `parse_dates`, `cap_outliers`, `no_action`).
  - `CHART_REGISTRY`: 7 Plotly chart renderers (`histogram`, `box`, `bar`, `line`, `scatter`, `heatmap`, `pie`) with deterministic dtype compatibility validation.
- **LangGraph Pipeline**:
  - Full pipeline wiring with linear stages and parallel fan-out via `Send()`.
  - Fan-in accumulator for `completed_charts` and `audit_log` using `operator.add`.
  - Human-in-the-loop approval gate before cleaning execution (`interrupt_before=["cleaning_exec"]`).
  - Two-layer self-check verification: code-level recomputation + LLM grounding check with automatic retry loop (max 2 retries) and `FLAGGED` status fallback.
- **OpenAI LLM Client**:
  - Centralized structured tool calling with retry and exponential backoff.
  - Support for custom model configuration via `OPENAI_MODEL` in `.env` (defaults to `gpt-5.4-mini`).
- **Streamlit Web UI**:
  - Live streaming progress display via `graph.stream()`.
  - Checkbox-based cleaning plan review and approval interface.
  - Dashboard with per-column data quality before/after comparisons, cleaning diffs, categorized visualization tabs with verification badges, and collapsible audit trail.
  - Session state caching keyed on uploaded file hash to prevent redundant LLM invocations on UI reruns.
- **Testing & Quality Assurance**:
  - Comprehensive pytest test suite with 64 tests across schemas, cleaning actions, and chart generation.
  - Full Ruff linting and formatting configuration (`ruff.toml`).
- **CI/CD**:
  - GitHub Actions CI workflow covering linting, multi-version Python testing (3.11, 3.12, 3.13), module import verification, and dependency security scanning (`ci.yml`).
  - Automated release workflow on tag push (`release.yml`).
