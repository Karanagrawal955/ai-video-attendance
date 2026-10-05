"""Encryption utilities for embeddings and sensitive data at rest."""

from __future__ import annotations

import base64
import os
import warnings
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import settings

# Fixed, public, NON-production key material.  Deliberately NOT the JWT
# secret: the JWT secret signs tokens and may be rotated/exposed without
# touching stored embeddings, and it is shipped as a default in .env.example.
_DEV_KEY_MATERIAL = "attendance-dev-only-key-do-not-use-in-production-v1"


def fernet_from_material(material: str) -> Fernet:
    """Derive the Fernet instance for a given key material string.

    Deterministic (fixed salt) so the same EMBEDDING_ENCRYPTION_KEY always
    decrypts the same rows.  Used by the key-rotation script with explicit
    old/new keys.
    """
    salt = b"attendance-salt-v1"  # Fixed salt for deterministic key derivation
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(material.encode()))
    return Fernet(key)


def _get_fernet() -> Fernet:
    """Key used for embeddings at rest.

    * `EMBEDDING_ENCRYPTION_KEY` is the only production source.  Production
      without it never reaches here: `app.config.Settings` raises at startup.
    * Non-production falls back to a fixed public constant so developers and
      tests work out of the box - never to the JWT secret.
    """
    material = settings.embedding_encryption_key
    if material is None:
        if settings.environment.strip().lower() in {"production", "prod"}:
            raise RuntimeError(
                "EMBEDDING_ENCRYPTION_KEY missing in production - refusing to "
                "derive an encryption key from anything else."
            )
        warnings.warn(
            "EMBEDDING_ENCRYPTION_KEY not set: using the built-in development key. "
            "Stored embeddings encrypted with it are NOT portable to production. "
            "Set a real key with `python scripts/generate_key.py`.",
            UserWarning,
            stacklevel=2,
        )
        material = _DEV_KEY_MATERIAL
    return fernet_from_material(material)


def active_key_material() -> str:
    """The key material currently used for embeddings at rest (see `_get_fernet`)."""
    if settings.embedding_encryption_key:
        return settings.embedding_encryption_key
    if settings.environment.strip().lower() in {"production", "prod"}:
        raise RuntimeError("EMBEDDING_ENCRYPTION_KEY missing in production")
    return _DEV_KEY_MATERIAL


_fernet = _get_fernet()


def encrypt_embeddings(embeddings: list[list[float]]) -> str:
    """Encrypt a list of embeddings (list of float lists) to a base64 string."""
    import json
    plaintext = json.dumps(embeddings).encode()
    return _fernet.encrypt(plaintext).decode()


def decrypt_embeddings(ciphertext: str) -> list[list[float]]:
    """Decrypt a base64 string back to list of embeddings."""
    import json
    plaintext = _fernet.decrypt(ciphertext.encode())
    return json.loads(plaintext.decode())


def encrypt_field(value: Any) -> str:
    """Encrypt any JSON-serializable value."""
    import json
    plaintext = json.dumps(value).encode()
    return _fernet.encrypt(plaintext).decode()


def decrypt_field(ciphertext: str) -> Any:
    """Decrypt an encrypted field."""
    import json
    plaintext = _fernet.decrypt(ciphertext.encode())
    return json.loads(plaintext.decode())


def generate_encryption_key() -> str:
    """Generate a new base64-encoded 32-byte key for EMBEDDING_ENCRYPTION_KEY."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode()