"""
Realtime event relay for WebSocket updates.

The FastAPI app creates a ConnectionManager in app.main and calls
set_connection_manager() during startup so other modules can emit events
without importing the FastAPI app (avoids circular imports).
"""

from typing import Optional, Dict, Any
import asyncio


_manager = None  # set by app.main


def set_connection_manager(manager_obj: Any) -> None:
    global _manager
    _manager = manager_obj


async def emit(session_id: str, event: str, data: Dict[str, Any]) -> None:
    """Emit a realtime event to a session WebSocket if connected."""
    if _manager is None or not session_id:
        return
    message = {"type": event, "data": data}
    try:
        await _manager.send_personal_message(message, session_id)
    except Exception:
        # Silently ignore WS failures to not impact workflow execution
        pass


