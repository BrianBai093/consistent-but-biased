# Consistent but Biased: chair personas in a simulated multi-agent surgical department

Code, prompts, run records and analysis outputs for the paper
"Consistent but Biased: Chair Personas Trade Noise for Bias in Multi-Agent Clinical Decisions"
(preprint: arXiv:2610.31325).

A simulated spine surgery department of seven LLM agents discussed 20 published cases of
degenerative cervical myelopathy from PMC-Patients. The statements that report each case and
assess its risks were generated once and replayed in every run. Only the persona of the chair
and its deputy varied: none (neutral), consultative, autocratic or avoidant, each written as
rules of conduct or as traits alone.

This is research software. It is not a medical device and must not be used for clinical decisions.

## Layout

| Path | Contents |
|---|---|
| `department/` | The simulated department: agents, LangGraph workflow, state |
| `department/prompts/` | All role prompts. Persona prefixes are `style_*.md` (chair) and `deputy_*.md` (deputy); suffix `_t` is the trait-only version, `_b` the reworded version b |
| `agents/`, `services/`, `*.py` | Shared schemas, model clients, case-report redaction and checks |
| `prompts/case_report_filter.md`, `prompts/flag_handling_judge.md` | The redaction prompt and the red-flag judge prompt |
| `data/test_sets/pmc_dcm_v1.json` | The 20 cases: PMC-Patients id, PMID, case group and the management reported in the full text |
| `runs/pmc-<id>/` | Per case: the pre-decision record (`case_preop.md`), the redaction review, and one JSON record per run (`arm-<arm>-attempt<n>.json`) |
| `runs/series-*/` | Trait-only ablation, replications (other model, version b) and separation experiments |
| `runs/flag_judgements/` | Red-flag codes from the judge model and the second judge |
| `analysis/` | Analysis outputs (JSON) and the two exploratory scripts |
| `tools/` | Experiment runners and the prespecified analyses |
| `tests/` | Offline tests (no model calls) |

Arm names in the code: `democratic` is the consultative persona in the paper, `director` is the chair and `vice_director` the deputy.

## Map from the paper to the files

| Paper | Run records | Analysis |
|---|---|---|
| Main analysis, rules version (H1-H5) | `runs/pmc-*/arm-*-attempt{1,2,3,4}.json` | `analysis/analysis_m3.json`, `tools/analyze_text_experiment.py` |
| Sensitivity analyses | same | `analysis/analysis_m3.json`, `analysis/mixed_model_m3.json`, `tools/fit_mixed_model.py` |
| Agreement against chance, Fleiss' kappa | same, plus `runs/series-traits` | `analysis/agreement_chance.json`, `analysis/agreement_chance.py` |
| Process measures | same | `analysis/process_measures.json`, `analysis/process_measures.py` |
| Trait-only ablation (H6) | `runs/series-traits` | `analysis/analysis_m9.json` |
| Red-flag handling (H7) | `runs/flag_judgements` | `analysis/analysis_flags.json`, `tools/judge_flag_handling.py` |
| Separation experiments (H8, H9) | `runs/series-chair-only`, `runs/series-deputy-only`, `runs/series-chair-alone` | `analysis/analysis_m10_m11.json`, `tools/analyze_chair_isolation.py` |
| Replications | `runs/series-model-openai`, `runs/series-wording-b` | `analysis/analysis_m5.json`, `analysis/analysis_m6.json`, `analysis/analysis_m3_subset.json` |

The case group of one case (C7, PMC-Patients 7440276-1) was corrected in the test set after some
analyses were run. This changes only the exploratory `by_group` block. `analysis/analysis_m3.json`
was regenerated with the corrected test set; the `by_group` blocks of the other analysis files
predate the correction.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest -q
```

Recompute the main analyses from the run records (no model calls):

```bash
uv run python -m tools.analyze_text_experiment --attempts 1,2,3,4
uv run python analysis/agreement_chance.py
uv run python analysis/process_measures.py
```

New runs call DeepSeek and OpenAI models and need API keys in `.env` (see `.env.example`), and
the PMC-Patients CSV for the redaction step:

```bash
uv run python -m tools.run_text_case_batch --help
```

The models were deepseek-v4-pro (discussion roles, redaction and red-flag judge) and
gpt-5.6-terra (radiologist, replication and second judge), with the providers' default sampling,
in October 2026. Model outputs are not deterministic, so new runs will differ from the records here.

The code also contains a pipeline for hospital cases with screenshots and images
(`app.py`, `graph.py`, `web/`). It was not used in the paper, and no hospital data is included.

## Licenses

- Code: MIT License (`LICENSE`).
- Case text and run records in `runs/` and `data/` derive from PMC-Patients and are shared under
  CC BY-NC-SA 4.0, the license of that dataset (`runs/LICENSE`).
- The OCR weights in `models/ocr/` keep their upstream license (see `models/ocr/SOURCES.md`).
