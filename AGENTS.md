# AGENTS.md — etisalat-use-case

IBM Telco churn challenge: churn classifier + extract-only LLM chat pipeline + FastAPI
service. Greenfield scaffold — most `src/telco_churn/` packages are `__init__.py` stubs;
the implementation plan lives in `openspec/changes/add-telco-churn-poc/`.

## Commands

```sh
# Test (must use this form — plain rtk/uv wrappers "collect 0 tests")
.venv/bin/python -m pytest -q

# Single test
.venv/bin/python -m pytest tests/unit/test_config.py -q

# Hooks (ruff → mypy → bandit; pip-audit on push, needs network)
uv run pre-commit run --all-files
uv run pre-commit install --install-hooks
```

- Python 3.13 via uv (`requires-python = ">=3.13"`). `uv sync` creates `.venv`.
- Editor config: `.vscode/settings.json` pins the interpreter to `.venv/bin/python`
  and adds `src/ to extraPaths — pytest's `pythonpath = ["src"]` does NOT
  cover the language server; without the extraPaths entry imports fail spuriously.

## Hard-won gotchas

- **Bare `AppConfig()` fails mypy** (`call-arg`): pydantic-settings populates
  required fields (e.g. `api_bearer_token`) from env at runtime. Annotate test
  instantiations with `# type: ignore[call-arg]  # pydantic-settings populates from env`
  (pattern already used in `config.py::get_settings`).
- **bandit B104** fires on `api_host = "0.0.0.0"` in `config.py` — intentional
  (container/serverless bind); it carries `# nosec B104` with justification.
  Don't remove either half.
- **pre-commit auto-fixes get rolled back** on commit failure ("Stashed changes
  conflicted"): stage files (`git add -A`) before running hooks so fixes stick.
- **pip-audit runs at pre-push and needs the network** (vuln DB). Offline push:
  `SKIP=pip-audit git push`.
- **gitleaks hook needs the gitleaks binary on PATH** (`uv tool install gitleaks`).
- `check-added-large-files --maxkb=800`: the challenge CSV is ~954 KB but lives
  in `data/`, which is excluded from hooks entirely.
- mypy hook uses a separate mirrored env with pinned `additional_dependencies`
  (see `.pre-commit-config.yaml`) — there's no installed typing package like
  `pydantic` there unless it's in that list.

## Repo conventions (differs from defaults)

- **Python-only repo**: no JS/TS, no shell, no SQL. Frontend/shell/sqlfluff
  hooks are deliberately commented out at the bottom of `.pre-commit-config.yaml`.
- **Markdown is not machine-formatted**: `openspec/` and `docs/` are excluded
  from all hooks; trailing double-space line breaks are preserved on other md files
  (`--markdown-linebreak-ext=md`). Edit specs/ADRs manually.
- Test staged as unit (`tests/unit/`), integration (`tests/integration/`),
  eval (`tests/eval/`); pytest addopts disable the cacheprovider.

## Domain language is canonical

`CONTEXT.md` is the glossary (Churn/Churn probability/Churner/Operating point/
Extract-only LLM/Feature request/Leakage guardrail/…). Use exactly one term per
concept in specs, code, and comments. Non-negotiables: the LLM never produces
numbers or predictions, only a schema-validated FeatureRequest; the classifier
computes Churn probability; published-band honesty (AUC 0.84–0.88) — scores
materially above trigger a leakage audit.

## Workflow

- **Feature-branch cycle (user-mandated)**: one branch per OpenSpec feature,
  named `<seq>/<feature-name>` (e.g. `p1/data-pipeline`, `p2/churn-model`,
  `p3/chat-pipeline`, `p4/chat-api`, `p5/deployment`). Cycle: implement ONLY
  that feature's tasks → test & verify fully → cleanup → commit & push →
  **wait for user merge** → next branch. `tasks.md` in the change directory
  tracks per-branch progress; a task is checked complete only after full
  verification.
- Spec-driven development via **OpenSpec**: change proposals/design/tasks under
  `openspec/changes/`, archived under archive when complete. Follow
  `openspec-*.md` skill instructions; plan against `openspec`, don't write
  standalone markdown TODO lists.
- ADRs in `docs/adr/` (Modal serverless over free-tier VMs; extract-only
  grammar-constrained LLM).
- Deployment: Modal (cloud) w/ FastAPI is the single service; LLM serving is
  runtime-agnostic (vLLM cloud / Ollama local); design.md forbids K8s/queues.
- Secrets: bearer token from env (`TELCO_API_BEARER_TOKEN`-style); never commit
  `.env` or tokens (gitleaks enforces).
