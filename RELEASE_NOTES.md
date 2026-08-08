# 🔍 AUDITA v1.0.0 — Official Release

**AUDITA** is an auditable, self-verifying data cleaning and visualization agent powered by LangGraph, OpenAI (`GPT-5.4-mini`), pandas, Plotly, and Streamlit.

---

### 🧠 Core Philosophy
> *"The LLM proposes. Code executes. Code verifies."*

AUDITA strictly prevents hallucinated or dangerous execution:
- The AI **never** writes or runs arbitrary Python/pandas/plotting code.
- All proposals are chosen from fixed enums and validated with **Pydantic schemas** against real dataset columns and dtypes before execution.
- All actions are executed by deterministic, tested registry functions.
- Every chart undergoes an **independent two-layer verification** (code recomputation + LLM grounding).
- Every event is recorded in an **immutable, append-only audit trail**.

---

### 🚀 Key Features

- **Automatic Ingestion**: Delimiter detection via `csv.Sniffer` and encoding fallback via `chardet`.
- **Quality Profiling**: Per-column audit computing missing rates, statistical distribution, IQR outliers, date detection, and fuzzy near-duplicate matching via `rapidfuzz`.
- **AI-Driven Cleaning Plan**: LLM proposes structured cleaning actions targeting real columns only.
- **Human-in-the-Loop Gate**: Checkbox-based interactive review allowing users to accept or reject individual cleaning steps prior to execution.
- **Deterministic Cleaning Registry (9 Handlers)**:
  - `impute_mean`, `impute_median`, `impute_mode`
  - `drop_rows`, `drop_column`
  - `standardize_categories`, `parse_dates`, `cap_outliers`, `no_action`
- **Before/After Diff Tracking**: Full tracking of modified rows, percentage improvements, and stat changes per action.
- **AI Insight & Visualization Planning**: Proposes ranked charts validated through deterministic dtype compatibility rules.
- **Deterministic Plotly Chart Registry (7 Renderers)**:
  - Histogram, Box Plot, Bar Chart, Line Chart, Scatter Plot, Heatmap, Pie Chart.
- **Two-Layer Self-Check Verification**:
  1. *Code recomputation*: Validates aggregated figures and flags misleading truncated axes.
  2. *LLM grounding*: Confirms the chart accurately represents the stated rationale.
  3. Automatic retry loop (max 2 retries) with a visible `⚠️ FLAGGED` fallback so visualizations are never silently dropped.
- **Streamlit Interactive UI**:
  - Live progress display through `graph.stream()`.
  - Caching by file hash to prevent redundant LLM invocations on UI reruns.
  - Verification badges (✅ Verified, ⚠️ Flagged, ❌ Failed).
  - Collapsible audit trail log.
- **OpenAI Model Flexibility**:
  - Configurable model selection via `OPENAI_MODEL` in `.env` (defaults to `GPT-5.4-mini`).

---

### 📦 Quickstart

```bash
# Clone the repository
git clone https://github.com/Aashutosh-Mahajan/AUDITA.git
cd AUDITA

# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# Launch the Streamlit dashboard
streamlit run audita/app.py
```

---

### 🧪 Test Suite & Quality Assurance

- **64 unit & integration tests** passing with 100% success rate across schemas, registries, and renderers.
- Linted and formatted with **Ruff** (0 errors).
- Automated CI pipeline testing across Python 3.11, 3.12, and 3.13.
