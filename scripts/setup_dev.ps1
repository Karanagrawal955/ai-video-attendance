<#
.SYNOPSIS
  One-command dev setup for a teammate on Windows.

  powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1

  Steps (7): python check -> venv -> pinned deps -> .env + encryption key
             -> DB migration -> model weights -> test suite.
  Everything is idempotent: re-running skips what already exists.

  NOTE: no Docker and no Redis are required for this path. The demo
  (scripts/run_demo.py) runs fully in-process; see README "Demo without
  Docker or Redis".
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot   # repo root = scripts/..
Set-Location $Root
Write-Host "repo root: $Root"

# ---------------------------------------------------------------- 1/7 python
Write-Host "`n== 1/7 python =="
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found on PATH - install Python 3.11+ first (https://python.org)"
}
$verText = & python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
$ver = [version]$verText
if ($ver -lt [version]"3.11") { throw "Python $verText found, need >= 3.11" }
Write-Host "python $verText  OK"

# ----------------------------------------------------------------- 2/7 venv
Write-Host "`n== 2/7 virtualenv =="
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    Write-Host "created .venv"
} else { Write-Host ".venv already exists" }
$VPy = ".venv\Scripts\python.exe"
& $VPy -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

# ---------------------------------------------------------- 3/7 dependencies
Write-Host "`n== 3/7 pinned dependencies (requirements.txt + requirements-dev.txt) =="
& $VPy -m pip install --quiet -r requirements.txt -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed - paste the log above; common fix: upgrade pip / check proxy"
}
& $VPy -c "import fastapi, sqlalchemy, insightface, cv2, pytest; print('imports OK')"

# ------------------------------------------------------------------ 4/7 .env
Write-Host "`n== 4/7 .env + encryption key =="
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "created .env from .env.example"
} else { Write-Host ".env already exists - untouched" }

$envContent = Get-Content .env -Raw
if ($envContent -match '(?m)^EMBEDDING_ENCRYPTION_KEY=\s*$') {
    $line = (& $VPy scripts\generate_key.py | Select-Object -Last 1)
    $key = ($line -replace '^EMBEDDING_ENCRYPTION_KEY=', '').Trim()
    if (-not $key) { throw "scripts/generate_key.py produced no key" }
    $envContent = $envContent -replace '(?m)^EMBEDDING_ENCRYPTION_KEY=.*$', "EMBEDDING_ENCRYPTION_KEY=$key"
    Set-Content -Path .env -Value $envContent -NoNewline
    Write-Host "generated EMBEDDING_ENCRYPTION_KEY (kept out of git: .env is gitignored)"
} else { Write-Host "EMBEDDING_ENCRYPTION_KEY already set - untouched" }

# ------------------------------------------------------------- 5/7 migration
Write-Host "`n== 5/7 database migration (alembic upgrade head) =="
$alembic = ".venv\Scripts\alembic.exe"
if (Test-Path $alembic) { & $alembic upgrade head }
else { & $VPy -m alembic upgrade head }
if ($LASTEXITCODE -ne 0) {
    throw "migration failed - check DATABASE_URL in .env (default is sqlite:///./data/attendance.db)"
}
Write-Host "schema is at head"

# -------------------------------------------------------- 6/7 model weights
Write-Host "`n== 6/7 model weights (scripts/download_models.py) =="
& $VPy scripts\download_models.py
if ($LASTEXITCODE -ne 0) {
    throw "model download failed - see the message above (weights are never committed)"
}

# ------------------------------------------------------------------ 7/7 tests
Write-Host "`n== 7/7 test suite =="
# NOTE: no extra -q here - pyproject addopts already supplies it, and a second
# one (-qq) suppresses the "N passed" summary line entirely.
& $VPy -m pytest
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

Write-Host "`nSETUP OK - next commands:"
Write-Host "  .venv\Scripts\python.exe scripts\run_demo.py --video <path-to-your-video.mp4>"
Write-Host "  .venv\Scripts\python.exe -m uvicorn app.main:app --reload"
Write-Host ""
Write-Host "  NOTE: no demo clip ships with this archive (media-free)."
Write-Host "  Build your own from enrolled photos:"
Write-Host "    .venv\Scripts\python.exe scripts\make_test_video.py --output eval\my_clip.mp4"
