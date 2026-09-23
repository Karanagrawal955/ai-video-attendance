"""WebSocket /ws/live-feed - real-time recognition events.

On connect: replay the newest 100 backlog events (stored in Redis), then
stream the pub/sub channel ``events:live``.  Clients may send
``{"type":"ping"}`` and receive ``{"type":"pong"}``.  Auth (when
AUTH_REQUIRED=true) is a ``?token=<jwt>`` query parameter.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..config import settings
from ..security import decode_token

logger = logging.getLogger("app.api.ws")

router = APIRouter()


@router.websocket("/ws/live-feed")
async def live_feed(
    websocket: WebSocket,
    token: str | None = Query(default=None),
) -> None:
    if settings.auth_required:
        try:
            decode_token(token or "")
        except Exception:  # noqa: BLE001 - HTTPException or otherwise
            await websocket.close(code=4401, reason="unauthorized")
            return
    await websocket.accept()

    import redis.asyncio as aioredis

    client = aioredis.from_url(
        settings.redis_url, decode_responses=True, socket_connect_timeout=3
    )
    pubsub = client.pubsub()

    try:
        # ---- replay backlog (index 0 = newest, send oldest first)
        try:
            backlog = await client.lrange(settings.events_recent_key, 0, 99)
            for raw in reversed(backlog):
                await websocket.send_text(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("backlog replay failed: %s", exc)

        await pubsub.subscribe(settings.events_channel)

        async def pump() -> None:
            async for message in pubsub.listen():
                if message.get("type") == "message":
                    await websocket.send_text(message["data"])

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                data = await websocket.receive_text()
                if "ping" in data[:40]:
                    await websocket.send_text('{"type":"pong"}')
        except WebSocketDisconnect:
            pass
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("websocket closed abnormally: %s", exc)
    finally:
        try:
            await pubsub.unsubscribe(settings.events_channel)
            await pubsub.aclose()
        except Exception:  # noqa: BLE001
            pass
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass
