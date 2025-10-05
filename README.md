# Agent Factory (MVP Bootstrap)

This is the **baby-step** scaffold to get us running on **Windows + VS Code** with a tiny FastAPI API, linting, tests, and CI.

> We'll iterate in small steps. This v0 includes: FastAPI app with `/health` and `/version`, pre-commit (Black + Ruff), pytest, and a GitHub Actions CI.

---

## 1) Clone & open in VS Code (Windows)

```powershell
# if you haven't already, clone your repo
git clone https://github.com/muhTamer/agent-factory.git
cd agent-factory

# copy the contents of this bootstrap into the repo folder
# (after downloading and unzipping, replace files as needed)
```

If you want to start fresh locally, you can also move these files into an empty folder and then link the remote (see step 5).

---

## 2) Create a Python 3.11 virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If `py -3.11` isn't available, use `py -3` or install Python 3.11 from python.org.

---

## 3) Pre-commit hooks (format & lint on commit)

```powershell
pre-commit install
# You can test hooks immediately:
pre-commit run --all-files
```

---

## 4) Run the dev server

```powershell
uvicorn app.main:app --reload
```

Now open http://127.0.0.1:8000/health and http://127.0.0.1:8000/version

---

## 5) Link to your GitHub repo

If this folder is not yet a git repo:

```powershell
git init
git add .
git commit -m "Bootstrap: FastAPI + pre-commit + CI"
git branch -M main
git remote add origin https://github.com/muhTamer/agent-factory.git
git push -u origin main
```

If it’s already cloned from GitHub, just commit and push:

```powershell
git add .
git commit -m "Bootstrap: FastAPI + pre-commit + CI"
git push
```

---

## 6) Run tests

```powershell
pytest -q
```

---

## 7) What’s included (v0)

- `app/main.py` — FastAPI app with `/health` and `/version`
- `tests/test_health.py` — smoke test
- `requirements.txt` — minimal deps for API + dev tooling
- `.pre-commit-config.yaml` — Black & Ruff
- `.github/workflows/ci.yml` — CI to check lint & tests
- `.vscode/` — opinions for formatting/on-save
- `data/` — placeholder folder (gitignored) for local-only files
- `.env` — not committed (use to keep local configs)

---

## 8) Next tiny step (proposed)

- Add a `/ingest` placeholder endpoint that validates CSV (BankFAQs.csv) and echoes basic stats — **no RAG yet**.
- Then wire a minimal embedding/index step.

We’ll proceed incrementally.
muhTamer