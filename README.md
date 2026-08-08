<p align="center">
  <h1 align="center">🔍 AUDITA</h1>
  <p align="center">
    <strong>Auditable, Self-Verifying Data Cleaning & Visualization Agent</strong>
  </p>
  <p align="center">
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-quickstart">Quickstart</a> •
    <a href="#-usage">Usage</a> •
    <a href="#-testing">Testing</a> •
    <a href="#-project-structure">Structure</a> •
    <a href="#-contributing">Contributing</a>
  </p>
  <p align="center">
    <img src="https://github.com/Aashutosh-Mahajan/AUDITA/actions/workflows/ci.yml/badge.svg" alt="CI">
    <img src="https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/LLM-Claude%20(Anthropic)-blueviolet" alt="LLM">
  </p>
</p>

---

## 🧠 Core Principle

> **The LLM proposes. Code executes. Code verifies.**

AUDITA is a data cleaning and visualization agent where the AI **never writes or executes code**. Instead:

1. The LLM proposes structured actions from fixed enums (cleaning actions, chart types)
2. Every proposal is **schema-validated** (Pydantic) against the real DataFrame's columns and dtypes
3. Hand-written, tested functions execute each action deterministically
4. A self-check layer **independently verifies** every chart through code recomputation + LLM grounding
5. Every decision — human or AI — is logged in an **immutable audit trail**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Auto-Ingest** | Uploads CSV with automatic delimiter detection (`csv.Sniffer`) and encoding fallback (`chardet`) |
| **Quality Audit** | Per-column profiling: missing %, outliers (IQR), fuzzy near-duplicate labels, date-parse detection |
| **AI Cleaning Plan** | LLM proposes cleaning actions from a fixed enum — validated against real columns before execution |
| **Human-in-the-Loop** | Review and approve/reject individual cleaning actions before they run |
| **Deterministic Cleaning** | 9 cleaning handlers (impute mean/median/mode, drop rows/column, standardize categories, parse dates, cap outliers) |
| **AI Visualization** | LLM proposes ranked visualizations — validated against dtype compatibility rules |
| **7 Chart Types** | Histogram, Box, Bar, Line, Scatter, Heatmap, Pie — all hand-written Plotly renderers |
| **Self-Verification** | 2-layer check: code recomputation (data accuracy, misleading axes) + LLM grounding (rationale support) |
| **Retry with Flagging** | Failed charts retry up to 2×, then render with a ⚠️ flag — never silently hidden |
| **Audit Trail** | Append-only log of every action (LLM proposals, code executions, rejections, verifications) |
| **Streaming Progress** | Live stage-by-stage updates in the Streamlit UI |
| **Result Caching** | File-hash caching — normal Streamlit reruns never re-invoke the pipeline |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI                              │
│  Upload → Progress → Cleaning Approval → Dashboard → Audit Log    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ graph.stream()
┌───────────────────────────────▼─────────────────────────────────────┐
│                       LANGGRAPH PIPELINE                            │
│                                                                     │
│  ┌────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Ingest │→ │Quality Audit│→ │Cleaning Plan │→ │ HUMAN GATE   │  │
│  │ (code) │  │   (code)    │  │   (LLM #1)   │  │ (interrupt)  │  │
│  └────────┘  └─────────────┘  └──────────────┘  └──────┬───────┘  │
│                                                         │          │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────────┐   │          │
│  │Cleaning Exec │← │Profiling │← │                 │←──┘          │
│  │   (code)     │→ │  (code)  │→ │Insight Planning │              │
│  └──────────────┘  └──────────┘  │    (LLM #2)     │              │
│                                  └────────┬────────┘              │
│                                           │ Send() fan-out        │
│                              ┌────────────┼────────────┐          │
│                              ▼            ▼            ▼          │
│                         ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│                         │ Chart   │ │ Chart   │ │ Chart   │      │
│                         │Builder 1│ │Builder 2│ │Builder N│      │
│                         └────┬────┘ └────┬────┘ └────┬────┘      │
│                              └────────────┼────────────┘          │
│                                           ▼ fan-in                │
│                                    ┌─────────────┐               │
│                                    │ Self-Check   │               │
│                                    │(code + LLM)  │──→ retry?    │
│                                    └──────┬───────┘               │
│                                           ▼                       │
│                                    ┌─────────────┐               │
│                                    │  Assemble   │               │
│                                    │   (code)    │               │
│                                    └─────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                          CORE LAYER                                 │
│  schemas.py │ cleaning_registry.py │ chart_registry.py │ llm_client│
│  (Pydantic) │  (9 handlers dict)   │ (7 renderers dict)│ (Anthropic│
│             │                      │ + dtype compat    │  + retry) │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **Fixed enum dispatch** | LLM can only propose from `CleaningActionType` (9 values) and `ChartType` (7 values) — no arbitrary code generation |
| **Column validation** | Every LLM-proposed column is checked against the real DataFrame before execution |
| **Dtype compatibility table** | Deterministic rules (e.g. scatter needs 2 numeric columns) — not an LLM judgment |
| **`operator.add` fan-in** | `completed_charts` and `audit_log` use LangGraph's `Annotated[list, operator.add]` for correct concurrent merge |
| **FLAGGED ≠ FAILED** | A flagged chart still renders with a warning — the user isn't silently missing content |
| **Batch grounding** | When >5 charts, grounding checks are batched into one LLM call for cost control |

---

## 🚀 Quickstart

### Prerequisites

- **Python 3.11+**
- **Anthropic API key** ([get one here](https://console.anthropic.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/Aashutosh-Mahajan/AUDITA.git
cd AUDITA

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-...
```

### Run the App

```bash
streamlit run audita/app.py
```

The app will open at `http://localhost:8501`. Upload a CSV file to start.

---

## 📖 Usage

### 1. Upload Data

Upload any CSV file via the sidebar. AUDITA automatically detects:
- **Delimiter** (comma, tab, semicolon, pipe) via `csv.Sniffer`
- **Encoding** (UTF-8, Latin-1, etc.) via `chardet`

### 2. Review Quality Audit

AUDITA profiles every column:
- Missing value percentage
- Numeric stats (mean, std, min, max, IQR outlier count)
- Categorical stats (top values, fuzzy near-duplicate labels)
- String columns that look like dates

### 3. Approve Cleaning Plan

The AI proposes cleaning actions — each with a rationale. You review with checkboxes:

| Action | Description |
|--------|-------------|
| `impute_mean` | Fill NaN with column mean |
| `impute_median` | Fill NaN with column median |
| `impute_mode` | Fill NaN with mode (any dtype) |
| `drop_rows` | Drop rows where column is NaN |
| `drop_column` | Remove the column entirely |
| `standardize_categories` | Lowercase + strip + fuzzy merge near-duplicates |
| `parse_dates` | Convert string column to datetime |
| `cap_outliers` | Cap values using IQR or percentile bounds |
| `no_action` | Skip this column |

### 4. View Dashboard

After cleaning, AUDITA generates:
- **Quality comparison** — before vs. after metrics per column
- **Cleaning diff** — what changed, how many rows affected
- **Visualizations** — grouped by category (distribution, relationship, trend, categorical)
- **Verification badges** — ✅ Verified, ⚠️ Flagged (with notes), ❌ Failed
- **Audit trail** — every LLM proposal, code execution, rejection, and verification

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest audita/tests/ -v

# Run with coverage
python -m pytest audita/tests/ -v --cov=audita --cov-report=term-missing

# Run specific test file
python -m pytest audita/tests/test_schemas.py -v
python -m pytest audita/tests/test_cleaning_registry.py -v
python -m pytest audita/tests/test_chart_registry.py -v
```

### Test Coverage

| Test File | Tests | What's Covered |
|-----------|-------|---------------|
| `test_schemas.py` | 25 | Enum membership, range validation (`priority_score` 0–5), required fields, invalid values |
| `test_cleaning_registry.py` | 17 | All 9 cleaning handlers against synthetic DataFrames with known values |
| `test_chart_registry.py` | 22 | All 7 chart renderers (compatible + incompatible dtypes), missing column handling |
| **Total** | **64** | — |

---

## 📁 Project Structure

```
AUDITA/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI (lint, test, security)
├── audita/
│   ├── app.py                      # Streamlit entrypoint
│   ├── core/
│   │   ├── schemas.py              # Pydantic models (8 models + 3 enums)
│   │   ├── cleaning_registry.py    # CleaningActionType → handler dispatch
│   │   ├── chart_registry.py       # ChartType → Plotly renderer dispatch
│   │   ├── audit_log.py            # Append-only audit trail helper
│   │   └── llm_client.py           # Anthropic API wrapper (structured output)
│   ├── graph/
│   │   ├── state.py                # PipelineState TypedDict
│   │   ├── build_graph.py          # LangGraph wiring (Send, fan-out/in, retry)
│   │   └── nodes/
│   │       ├── ingest.py           # CSV ingest (auto-delimiter, encoding)
│   │       ├── quality_audit.py    # Per-column profiling (pure code)
│   │       ├── cleaning_plan.py    # LLM call #1 (propose cleaning)
│   │       ├── cleaning_exec.py    # Dispatch cleaning + diff tracking
│   │       ├── profiling.py        # Post-cleaning profile
│   │       ├── insight_planning.py # LLM call #2 (propose visualizations)
│   │       ├── chart_builder.py    # Fan-out target (render charts)
│   │       ├── self_check.py       # Code verify + LLM grounding
│   │       └── assemble.py         # Final dashboard assembly
│   ├── ui/
│   │   ├── components.py           # Badges, cards, approval widget
│   │   └── sections.py             # 4 dashboard section renderers
│   └── tests/
│       ├── test_schemas.py         # Schema validation tests
│       ├── test_cleaning_registry.py # Cleaning handler tests
│       └── test_chart_registry.py  # Chart renderer tests
├── .env.example                    # ANTHROPIC_API_KEY=
├── .gitignore
├── requirements.txt
├── AUDITA_BUILD_SPEC.md            # Full build specification
└── README.md                       # This file
```

---

## ⚙️ CI/CD

GitHub Actions runs on every push to `main`/`develop` and every pull request:

| Job | What it Does |
|-----|-------------|
| **Lint** | `ruff check` + `ruff format --check` for code quality |
| **Test** | Full test suite across Python 3.11, 3.12, 3.13 with coverage |
| **Import Check** | Verifies all core modules import cleanly |
| **Security** | `pip-audit` scans dependencies for known vulnerabilities |

---

## 🛡️ Security

- **API keys** are loaded from `.env` (never committed — in `.gitignore`)
- The LLM **never** receives the raw DataFrame — only the quality audit dict
- All LLM output is **schema-validated** before execution
- No arbitrary code execution — all actions are dispatched from fixed registries
- `pip-audit` runs in CI to catch vulnerable dependencies

---

## 🗺️ Roadmap (Explicit Non-Goals for v1)

These are intentionally deferred, not forgotten:

- [ ] Arbitrary LLM-generated plotting/cleaning code
- [ ] Multi-file upload / join inference
- [ ] Live database connections or streaming data
- [ ] Per-chart natural-language follow-up / targeted re-analysis

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`python -m pytest audita/tests/ -v`)
5. Run lint (`ruff check audita/`)
6. Commit with [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `test:`, `docs:`)
7. Open a pull request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Anthropic Claude](https://www.anthropic.com/) — structured output for cleaning/visualization proposals
- [LangGraph](https://github.com/langchain-ai/langgraph) — pipeline orchestration with fan-out/fan-in
- [Plotly](https://plotly.com/) — interactive charting
- [Streamlit](https://streamlit.io/) — dashboard UI
- [Pydantic](https://docs.pydantic.dev/) — schema validation
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) — fuzzy string matching for duplicate detection
