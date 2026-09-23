"""End-to-end smoke test against a running API (no redis/GPU strictly needed).

    python scripts/smoke_test.py http://127.0.0.1:8000

Checks, in order:
  metadata, health, auth gate (401), token issue (bad + good), student CRUD,
  camera CRUD + start/stop, attendance queries, gpu-status shape, docs.

Calls that *require* redis or the inference service PASS on a graceful 503
(service clearly reported as unavailable) when those services are down -
everything else must return its exact success status.  Exit code 0 = pass.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

PASS, FAIL, NOTE = "PASS", "FAIL", "NOTE"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", *, graceful: int | None = None,
          status: int | None = None) -> None:
    """Record a check. `graceful` = an acceptable degraded status (e.g. 503)."""
    if ok:
        verdict, msg = PASS, detail
    elif graceful is not None and status == graceful:
        verdict, msg = NOTE, f"degraded {status}: {detail}"
    else:
        verdict, msg = FAIL, f"status={status} {detail}"
    _results.append((verdict, name, msg))
    print(f"[{verdict}] {name}" + (f" - {msg}" if msg else ""))


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    user = os.environ.get("ADMIN_USERNAME", "admin")
    password = os.environ.get("ADMIN_PASSWORD", "admin")
    c = httpx.Client(base_url=base, timeout=20.0)

    # Wait for server readiness (~15s) so the script can be run right after
    # starting uvicorn.
    ready = False
    for _ in range(30):
        try:
            if c.get("/", timeout=3.0).status_code == 200:
                ready = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    if not ready:
        print(f"[FAIL] server not reachable at {base}")
        return 1

    # ------------------------------------------------------------- metadata
    r = c.get("/")
    check("GET / metadata", r.status_code == 200 and r.json().get("docs") == "/docs",
          status=r.status_code)

    r = c.get("/system/health")
    body = r.json() if r.status_code == 200 else {}
    check("GET /system/health", r.status_code == 200 and
          body.get("checks", {}).get("database") == "up",
          f"checks={body.get('checks')}", status=r.status_code)

    # ----------------------------------------------------------------- auth
    r = c.get("/students")
    check("unauthenticated request rejected", r.status_code == 401,
          status=r.status_code)

    r = c.post("/auth/token", json={"username": user, "password": "wrong"})
    check("bad credentials rejected", r.status_code == 401, status=r.status_code)

    r = c.post("/auth/token", json={"username": user, "password": password})
    token = r.json().get("access_token") if r.status_code == 200 else None
    check("POST /auth/token issues JWT", bool(token), r.text[:120],
          status=r.status_code)
    if not token:
        return _finish()
    h = {"Authorization": f"Bearer {token}"}

    r = c.get("/students", headers=h)
    check("GET /students", r.status_code == 200, r.text[:120], status=r.status_code)

    # ------------------------------------------------------------- students
    photos = [
        ("photos", (f"p{i}.jpg", b"\xff\xd8\xff\xe0smoke" + bytes([i]), "image/jpeg"))
        for i in range(3)
    ]
    r = c.post(
        "/students",
        data={"name": "Smoke Test", "registration_no": "SMOKE-1"},
        files=photos,
        headers=h,
    )
    enrolled = r.status_code == 201
    check("POST /students (enroll)", enrolled,
          r.json().get("detail", "") if isinstance(r.json(), dict) else r.text[:120],
          graceful=503, status=r.status_code)
    if not enrolled:
        print("       (503 = inference service down; enrollment needs it - "
              "start `docker compose up -d inference`)")

    sid = r.json().get("id") if enrolled else None
    if sid:
        r = c.get(f"/students/{sid}", headers=h)
        check("GET /students/{id}", r.status_code == 200 and
              r.json()["embedding_count"] == 3, status=r.status_code)
        r = c.put(f"/students/{sid}", json={"section": "CSE-SMOKE"}, headers=h)
        check("PUT /students/{id}", r.status_code == 200 and
              r.json()["section"] == "CSE-SMOKE", status=r.status_code)
        r = c.delete(f"/students/{sid}", headers=h)
        check("DELETE /students/{id}", r.status_code == 200, status=r.status_code)
        r = c.get(f"/students/{sid}", headers=h)
        check("student gone after delete", r.status_code == 404, status=r.status_code)

    r = c.get("/students/999999", headers=h)
    check("GET unknown student 404", r.status_code == 404, status=r.status_code)

    # -------------------------------------------------------------- cameras
    r = c.post("/cameras", json={"name": "Smoke Cam", "type": "entry",
                                 "file_path": "/samples/videos/none.mp4",
                                 "sampling_rate": 5}, headers=h)
    cam_id = r.json().get("id") if r.status_code == 201 else None
    check("POST /cameras", cam_id is not None, r.text[:120], status=r.status_code)

    r = c.get("/cameras", headers=h)
    check("GET /cameras", r.status_code == 200, status=r.status_code)

    if cam_id:
        r = c.put(f"/cameras/{cam_id}", json={"sampling_rate": 10}, headers=h)
        check("PUT /cameras/{id}", r.status_code == 200 and
              r.json()["sampling_rate"] == 10, status=r.status_code)

        r = c.post(f"/cameras/{cam_id}/start", headers=h)
        check("POST /cameras/{id}/start", r.status_code == 200,
              "started (slot claimed)" if r.status_code == 200
              else "needs redis+worker",
              graceful=503, status=r.status_code)
        if r.status_code == 200:
            rr = c.post(f"/cameras/{cam_id}/stop?force=true", headers=h)
            check("POST /cameras/{id}/stop?force", rr.status_code == 200 and
                  rr.json()["state"] == "stopped", rr.text[:120],
                  status=rr.status_code)

        r = c.delete(f"/cameras/{cam_id}", headers=h)
        check("DELETE /cameras/{id}", r.status_code == 200, r.text[:120],
              graceful=409, status=r.status_code)

    # ----------------------------------------------------------- attendance
    r = c.get("/attendance/summary", headers=h)
    check("GET /attendance/summary", r.status_code == 200 and
          "totals" in r.json(), r.text[:120], status=r.status_code)
    r = c.get("/attendance/summary?date=not-a-date", headers=h)
    check("bad date rejected 422", r.status_code == 422, status=r.status_code)
    r = c.get("/attendance/live", headers=h)
    check("GET /attendance/live", r.status_code == 200, status=r.status_code)
    r = c.get("/attendance/logs?limit=5", headers=h)
    check("GET /attendance/logs", r.status_code == 200 and
          "items" in r.json(), status=r.status_code)
    r = c.get("/attendance/999999", headers=h)
    check("GET unknown student attendance 404", r.status_code == 404,
          status=r.status_code)

    # --------------------------------------------------------------- system
    r = c.get("/system/gpu-status", headers=h)
    body = r.json() if r.status_code == 200 else {}
    check("GET /system/gpu-status", r.status_code == 200 and
          body.get("service") in ("online", "offline"),
          f"service={body.get('service')} provider={body.get('provider')}",
          status=r.status_code)

    # ----------------------------------------------------------------- docs
    r = c.get("/docs")
    check("GET /docs", r.status_code == 200, status=r.status_code)
    r = c.get("/openapi.json")
    spec = r.json() if r.status_code == 200 else {}
    n_paths = len(spec.get("paths", {}))
    check("GET /openapi.json", r.status_code == 200 and n_paths >= 15,
          f"{n_paths} paths", status=r.status_code)

    return _finish()


def _finish() -> int:
    failed = [x for x in _results if x[0] == FAIL]
    notes = [x for x in _results if x[0] == NOTE]
    print(f"\n{len(_results)} checks: {len(_results) - len(failed) - len(notes)} "
          f"passed, {len(notes)} degraded, {len(failed)} failed")
    for _, name, msg in failed:
        print(f"  FAILED: {name} ({msg})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
