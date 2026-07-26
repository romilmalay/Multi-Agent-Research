# Multi-Agent Research System

A 7-agent / 8-node LangGraph pipeline that turns a research question into a cited,
peer-reviewed report, served as a job by a FastAPI + Postgres backend with a React UI.

- `plan.md` — what we build, phase by phase
- `docs/explained.md` — plain-language explanation
- `docs/reference_plan.md` — rationale, rejected options

## Layout

```
backend/    research_system/ (the agent, no FastAPI/DB) + app/ (the service)
frontend/   React + Vite + TypeScript
```

`app` may import `research_system`. Never the reverse — enforced by import-linter.

## Develop

```bash
make install          # uv sync + pre-commit hooks
make lint type test   # must be green before a phase is done
```

Copy `backend/.env.example` to `backend/.env` and fill in the API keys.

## Status

Phase 0 (toolchain and scaffold) complete. See the progress list in `plan.md`.
