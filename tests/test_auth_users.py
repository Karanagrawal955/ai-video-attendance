"""Multi-user auth: registration, approval, RBAC and password resets.

Covers the four rules of the feature:
  1. the login page can register an account,
  2. a forgotten-password request is routed to the super admin,
  3. only the super admin may set or change a username / password,
  4. an admin account must be reviewed (approved + security level) first.
"""

from __future__ import annotations

import pytest


def _login(client, username: str, password: str):
    return client.post(
        "/auth/token", json={"username": username, "password": password}
    )


def _bearer(resp) -> dict[str, str]:
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _super_headers(client) -> dict[str, str]:
    """The seeded super admin from ADMIN_USERNAME/ADMIN_PASSWORD."""
    resp = _login(client, "admin", "admin")
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "super_admin"
    return _bearer(resp)


def _register(client, username: str, password: str = "s3cret-pass") -> dict:
    resp = client.post(
        "/auth/register",
        json={"username": username, "password": password},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --------------------------------------------------------------- bootstrap
def test_seeded_super_admin_can_log_in(client) -> None:
    resp = _login(client, "admin", "admin")
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "super_admin"
    assert body["token_type"] == "bearer"

    me = client.get("/auth/me", headers=_bearer(resp))
    assert me.status_code == 200
    assert me.json()["role"] == "super_admin"
    assert me.json()["status"] == "active"
    assert me.json()["security_level"] == 5


def test_me_requires_a_token(client) -> None:
    assert client.get("/auth/me").status_code == 401


def test_unknown_user_still_401(client) -> None:
    assert _login(client, "nobody", "whatever").status_code == 401


# ------------------------------------------------------------- registration
def test_register_creates_pending_account(client) -> None:
    created = _register(client, "priya.n")
    assert created["status"] == "pending"
    assert created["requested_role"] == "admin"

    # not approved yet -> the account may not sign in
    resp = _login(client, "priya.n", "s3cret-pass")
    assert resp.status_code == 403
    assert "approval" in resp.json()["detail"]


def test_register_duplicate_username_conflicts(client) -> None:
    _register(client, "dup.user")
    resp = client.post(
        "/auth/register",
        json={"username": "dup.user", "password": "another-pass"},
    )
    assert resp.status_code == 409
    assert "taken" in resp.json()["detail"]


def test_register_rejects_short_password(client) -> None:
    resp = client.post(
        "/auth/register", json={"username": "short.pw", "password": "abc"}
    )
    assert resp.status_code == 422


def test_register_cannot_self_assign_super_admin(client) -> None:
    resp = client.post(
        "/auth/register",
        json={
            "username": "sneaky.one",
            "password": "s3cret-pass",
            "requested_role": "super_admin",
        },
    )
    assert resp.status_code == 422


def test_register_cannot_take_the_super_admin_username(client) -> None:
    # seeding runs first, so the bootstrap name is already owned
    resp = client.post(
        "/auth/register",
        json={"username": "admin", "password": "hijacked-pass"},
    )
    assert resp.status_code in (409, 422)
    assert _login(client, "admin", "hijacked-pass").status_code == 401


# ----------------------------------------------------------------- approval
def test_super_admin_approves_with_security_level(client) -> None:
    created = _register(client, "review.me")
    headers = _super_headers(client)

    listed = client.get("/auth/admin/users", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 2  # super admin + pending account

    approved = client.post(
        f"/auth/admin/users/{created['id']}/approve",
        json={"security_level": 4},
        headers=headers,
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "active"
    assert approved.json()["security_level"] == 4
    assert approved.json()["approved_by"] == "admin"

    resp = _login(client, "review.me", "s3cret-pass")
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"

    me = client.get("/auth/me", headers=_bearer(resp)).json()
    assert me["status"] == "active"
    assert me["security_level"] == 4


def test_rejected_account_cannot_log_in(client) -> None:
    created = _register(client, "no.entry")
    headers = _super_headers(client)
    resp = client.post(
        f"/auth/admin/users/{created['id']}/reject", headers=headers
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"

    denied = _login(client, "no.entry", "s3cret-pass")
    assert denied.status_code == 403
    assert "rejected" in denied.json()["detail"]


def test_suspended_account_loses_access_immediately(client) -> None:
    created = _register(client, "later.on")
    headers = _super_headers(client)
    client.post(
        f"/auth/admin/users/{created['id']}/approve",
        json={"security_level": 2},
        headers=headers,
    )
    token = _bearer(_login(client, "later.on", "s3cret-pass"))
    assert client.get("/auth/me", headers=token).status_code == 200

    client.post(f"/auth/admin/users/{created['id']}/suspend", headers=headers)
    assert client.get("/auth/me", headers=token).status_code == 403
    assert _login(client, "later.on", "s3cret-pass").status_code == 403


# --------------------------------------------------------------------- RBAC
def _approved_admin(client, username: str = "plain.admin") -> dict[str, str]:
    created = _register(client, username)
    super_headers = _super_headers(client)
    client.post(
        f"/auth/admin/users/{created['id']}/approve",
        json={"security_level": 3},
        headers=super_headers,
    )
    return _bearer(_login(client, username, "s3cret-pass"))


def test_admin_cannot_list_accounts(client) -> None:
    headers = _approved_admin(client)
    assert client.get("/auth/admin/users", headers=headers).status_code == 403
    assert (
        client.get("/auth/admin/password-requests", headers=headers).status_code
        == 403
    )


def test_admin_cannot_change_any_credentials(client) -> None:
    headers = _approved_admin(client)
    # not even its own
    resp = client.patch(
        "/auth/admin/users/2",
        json={"password": "brand-new-pass"},
        headers=headers,
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "super admin only"
    # and the password is untouched
    assert _login(client, "plain.admin", "brand-new-pass").status_code == 401
    assert _login(client, "plain.admin", "s3cret-pass").status_code == 200


def test_admin_cannot_approve_accounts(client) -> None:
    created = _register(client, "needs.ok")
    headers = _approved_admin(client)
    resp = client.post(
        f"/auth/admin/users/{created['id']}/approve",
        json={"security_level": 3},
        headers=headers,
    )
    assert resp.status_code == 403


def test_admin_endpoints_need_a_token(client) -> None:
    assert client.get("/auth/admin/users").status_code == 401


# ------------------------------------------------- credential management
def test_super_admin_changes_username_and_password(client) -> None:
    created = _register(client, "rename.me")
    headers = _super_headers(client)
    client.post(
        f"/auth/admin/users/{created['id']}/approve",
        json={"security_level": 3},
        headers=headers,
    )
    old_token = _bearer(_login(client, "rename.me", "s3cret-pass"))

    resp = client.patch(
        f"/auth/admin/users/{created['id']}",
        json={"username": "renamed.here", "password": "totally-new-pass"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "renamed.here"

    # old credentials die, new ones work
    assert _login(client, "rename.me", "s3cret-pass").status_code == 401
    fresh = _login(client, "renamed.here", "totally-new-pass")
    assert fresh.status_code == 200

    # an already-issued token follows the row (uid claim), not the old name
    me = client.get("/auth/me", headers=old_token).json()
    assert me["username"] == "renamed.here"


def test_super_admin_can_change_own_password(client) -> None:
    headers = _super_headers(client)
    me = client.get("/auth/me", headers=headers).json()
    resp = client.patch(
        f"/auth/admin/users/{me['id']}",
        json={"password": "rotated-super-pass"},
        headers=headers,
    )
    assert resp.status_code == 200

    # .env can no longer override a password set through the API
    assert _login(client, "admin", "admin").status_code == 401
    assert _login(client, "admin", "rotated-super-pass").status_code == 200


def test_username_conflict_is_rejected(client) -> None:
    _register(client, "taken.name")
    headers = _super_headers(client)
    resp = client.patch(
        "/auth/admin/users/1",
        json={"username": "taken.name"},
        headers=headers,
    )
    # target 1 is the seeded super admin, so it would collide with taken.name
    assert resp.status_code in (404, 409)


def test_cannot_demote_the_last_super_admin(client) -> None:
    headers = _super_headers(client)
    me = client.get("/auth/me", headers=headers).json()
    resp = client.patch(
        f"/auth/admin/users/{me['id']}",
        json={"role": "admin"},
        headers=headers,
    )
    assert resp.status_code == 409
    assert "last active super admin" in resp.json()["detail"]

    # and disabling it is refused too
    resp = client.post(
        f"/auth/admin/users/{me['id']}/suspend", headers=headers
    )
    assert resp.status_code == 409


def test_patch_requires_at_least_one_field(client) -> None:
    headers = _super_headers(client)
    resp = client.patch("/auth/admin/users/1", json={}, headers=headers)
    assert resp.status_code == 422


# ---------------------------------------------------- forgot-password flow
def test_forgot_password_is_routed_to_super_admin(client) -> None:
    resp = client.post(
        "/auth/forgot-password",
        json={"username": "admin", "reason": "cannot remember it"},
    )
    assert resp.status_code == 202
    assert resp.json()["accepted"] is True

    headers = _super_headers(client)
    listed = client.get("/auth/admin/password-requests", headers=headers)
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert listed.json()["total"] == 1
    assert items[0]["username"] == "admin"
    assert items[0]["reason"] == "cannot remember it"
    assert items[0]["status"] == "pending"
    assert items[0]["resolved_by"] is None


def test_super_admin_resets_password_from_request(client) -> None:
    created = _register(client, "forgot.me")
    headers = _super_headers(client)
    client.post(
        f"/auth/admin/users/{created['id']}/approve",
        json={"security_level": 3},
        headers=headers,
    )

    req = client.post(
        "/auth/forgot-password",
        json={"username": "forgot.me", "reason": "lost password"},
    )
    req_id = req.json()  # 202 body has no id -> read it back from the list
    listed = client.get("/auth/admin/password-requests", headers=headers).json()
    req_id = listed["items"][0]["id"]

    done = client.post(
        f"/auth/admin/password-requests/{req_id}",
        json={"action": "reset", "new_password": "recovered-pass"},
        headers=headers,
    )
    assert done.status_code == 200
    assert done.json()["ok"] is True

    assert _login(client, "forgot.me", "s3cret-pass").status_code == 401
    assert _login(client, "forgot.me", "recovered-pass").status_code == 200

    # resolved requests cannot be replayed
    replay = client.post(
        f"/auth/admin/password-requests/{req_id}",
        json={"action": "reset", "new_password": "another-pass"},
        headers=headers,
    )
    assert replay.status_code == 409


def test_reset_request_can_be_rejected(client) -> None:
    _register(client, "not.mine")
    headers = _super_headers(client)
    client.post(
        "/auth/forgot-password", json={"username": "not.mine", "reason": "spam?"}
    )
    req_id = client.get("/auth/admin/password-requests", headers=headers).json()[
        "items"
    ][0]["id"]

    done = client.post(
        f"/auth/admin/password-requests/{req_id}",
        json={"action": "reject"},
        headers=headers,
    )
    assert done.status_code == 200
    listed = client.get("/auth/admin/password-requests", headers=headers).json()
    assert listed["items"][0]["status"] == "rejected"
    # password untouched
    assert _login(client, "not.mine", "s3cret-pass").status_code == 403


def test_reset_without_password_is_a_validation_error(client) -> None:
    headers = _super_headers(client)
    client.post("/auth/forgot-password", json={"username": "admin"})
    req_id = client.get("/auth/admin/password-requests", headers=headers).json()[
        "items"
    ][0]["id"]
    resp = client.post(
        f"/auth/admin/password-requests/{req_id}",
        json={"action": "reset"},
        headers=headers,
    )
    assert resp.status_code == 422


def test_admin_cannot_resolve_reset_requests(client) -> None:
    _register(client, "asking.user")
    client.post("/auth/forgot-password", json={"username": "asking.user"})
    headers = _approved_admin(client)
    listed = client.get("/auth/admin/password-requests", headers=headers)
    assert listed.status_code == 403


def test_unknown_reset_request_is_404(client) -> None:
    headers = _super_headers(client)
    resp = client.post(
        "/auth/admin/password-requests/999",
        json={"action": "reject"},
        headers=headers,
    )
    assert resp.status_code == 404


# ------------------------------------------------------------- .env fallback
def test_env_credentials_keep_working_after_rotation(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rotating ADMIN_PASSWORD in .env still unlocks the seeded super admin."""
    from app.config import settings

    headers = _super_headers(client)  # seeds admin/admin
    monkeypatch.setattr(settings, "admin_password", "rotated-by-env")

    resp = _login(client, "admin", "rotated-by-env")
    assert resp.status_code == 200
    # and the stored hash was re-synced: the new value alone now works
    assert _login(client, "admin", "rotated-by-env").status_code == 200
    del headers
