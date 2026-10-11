# ml_project

MLOps project test

## Structure

- `app/` – serving / API code
- `training/` – training scripts and pipelines
- `tests/` – tests
- `notebooks/` – exploration
- `data/` – datasets (not tracked by git)
- `models/` – saved model artifacts (not tracked by git)
- `mlruns/` – MLflow tracking (not tracked by git)
- `logs/` – logs (not tracked by git)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
