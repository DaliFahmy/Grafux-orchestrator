from __future__ import annotations

import time
import uuid
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("sandbox.e2b")


class E2BClient:
    """Wrapper around the E2B Code Interpreter SDK."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.e2b_api_key
        self._default_timeout = settings.sandbox_timeout

    async def execute_code(
        self,
        code: str,
        execution_id: str = "",
        timeout: int | None = None,
        packages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run Python code in an isolated E2B sandbox.

        Returns a dict with stdout, stderr, exit_code, and session_id.
        """
        if not self._api_key:
            log.warning("e2b_api_key_not_configured")
            return {
                "session_id": "local",
                "stdout": "",
                "stderr": "E2B API key not configured",
                "exit_code": 1,
                "error": "E2B_API_KEY not set",
            }

        from app.modules.streaming.event_bus import emit
        from app.shared.events import SANDBOX_STDERR, SANDBOX_STDOUT

        session_id = str(uuid.uuid4())
        effective_timeout = timeout or self._default_timeout
        start = time.monotonic()

        try:
            from e2b_code_interpreter import Sandbox

            sbx = await Sandbox.create_async(
                api_key=self._api_key,
                timeout=effective_timeout,
            )

            # Install required packages
            if packages:
                install_code = f"import subprocess; subprocess.run(['pip', 'install', '-q', {', '.join(repr(p) for p in packages)}])"
                await sbx.run_code(install_code)

            execution = await sbx.run_code(code)

            stdout = "\n".join(
                r.text for r in execution.logs.stdout if r.text
            )
            stderr = "\n".join(
                r.text for r in execution.logs.stderr if r.text
            )

            if execution_id:
                if stdout:
                    await emit(execution_id, SANDBOX_STDOUT, {"text": stdout[:2000]})
                if stderr:
                    await emit(execution_id, SANDBOX_STDERR, {"text": stderr[:2000]})

            await sbx.kill()

            result = {
                "session_id": session_id,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": 0 if not execution.error else 1,
                "error": str(execution.error) if execution.error else None,
                "duration_ms": int((time.monotonic() - start) * 1000),
            }

            # Persist session record
            await self._persist_session(execution_id, session_id, result)
            return result

        except Exception as exc:
            log.error("e2b_execution_failed", error=str(exc), execution_id=execution_id)
            return {
                "session_id": session_id,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": 1,
                "error": str(exc),
                "duration_ms": int((time.monotonic() - start) * 1000),
            }

    async def _persist_session(
        self, execution_id: str, session_id: str, result: dict[str, Any]
    ) -> None:
        if not execution_id:
            return
        try:
            from app.core.database import get_db_session
            from app.modules.persistence.models import SandboxSession
            async with get_db_session() as db:
                db.add(SandboxSession(
                    execution_id=execution_id,
                    e2b_session_id=session_id,
                    status="complete",
                    output=result.get("stdout", ""),
                    error_output=result.get("stderr", ""),
                    exit_code=result.get("exit_code"),
                ))
                await db.commit()
        except Exception as exc:
            log.warning("sandbox_persist_failed", error=str(exc))
