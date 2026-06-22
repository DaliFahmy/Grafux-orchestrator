from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.core.logging import get_logger
from app.modules.agent.runtime import AgentRuntime
from app.modules.agent.schemas import AgentConfig, MultiAgentConfig
from app.modules.streaming.event_bus import emit, emit_agent_thought
from app.shared.events import AGENT_MESSAGE

log = get_logger("agent.coordinator")


class MultiAgentCoordinator:
    """Orchestrates multiple AI agents working in parallel or sequence."""

    async def run(self, config: MultiAgentConfig) -> dict[str, Any]:
        """Execute a multi-agent workflow.

        The coordinator LLM decomposes the task, assigns sub-tasks to
        specialist agents, collects results, and synthesises a final answer.
        """
        settings = get_settings()
        execution_id = config.execution_id

        await emit_agent_thought(
            execution_id, "coordinator", f"Decomposing task: {config.task[:200]}"
        )

        # Step 1: Decompose task using coordinator LLM
        sub_tasks = await self._decompose_task(config.task, config.agents, settings)
        log.info("task_decomposed", execution_id=execution_id, sub_tasks=len(sub_tasks))

        # Step 2: Run agents in parallel for their sub-tasks
        import asyncio
        runtime = AgentRuntime()
        tasks = []
        agent_map = {a.agent_id: a for a in config.agents}

        for sub_task in sub_tasks:
            agent_id = sub_task.get("agent_id")
            if not agent_id or agent_id not in agent_map:
                continue
            agent_config = agent_map[agent_id]
            messages = [{"role": "user", "content": sub_task["task"]}]
            tasks.append(runtime.run(
                config=agent_config,
                messages=messages,
                execution_id=execution_id,
                org_id=config.org_id,
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Synthesise results
        outputs = []
        for sub_task, result in zip(sub_tasks, results, strict=False):
            if isinstance(result, Exception):
                outputs.append(f"Agent {sub_task.get('agent_id')}: ERROR — {result}")
            else:
                outputs.append(
                    f"Agent {sub_task.get('agent_id')}: {result.get('output', '')[:1000]}"
                )

        synthesis = await self._synthesise(config.task, outputs, settings)

        await emit(execution_id, AGENT_MESSAGE, {
            "agent_id": "coordinator",
            "content": synthesis,
        })

        return {
            "output": synthesis,
            "agent_results": outputs,
            "sub_tasks": sub_tasks,
        }

    async def _decompose_task(
        self,
        task: str,
        agents: list[AgentConfig],
        settings: Any,
    ) -> list[dict[str, Any]]:
        if not settings.openai_api_key:
            return [{"agent_id": agents[0].agent_id if agents else "agent", "task": task}]

        import json

        from app.core.http_client import get_http_client
        agent_descriptions = "\n".join(
            f"- {a.agent_id}: {a.name} — {a.system_prompt[:100]}"
            for a in agents
        )
        prompt = (
            f"You are a task coordinator. Decompose this task into sub-tasks for specialists.\n\n"
            f"Task: {task}\n\nAvailable agents:\n{agent_descriptions}\n\n"
            "Return a JSON array: [{\"agent_id\": \"...\", \"task\": \"...\"}]"
        )
        try:
            resp = await get_http_client().post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "response_format": {"type": "json_object"},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(content)
                return data if isinstance(data, list) else data.get("tasks", [])
        except Exception as exc:
            log.warning("task_decomposition_failed", error=str(exc))
        return [{"agent_id": agents[0].agent_id if agents else "agent", "task": task}]

    async def _synthesise(
        self,
        original_task: str,
        agent_outputs: list[str],
        settings: Any,
    ) -> str:
        if not settings.openai_api_key:
            return "\n\n".join(agent_outputs)

        from app.core.http_client import get_http_client
        combined = "\n\n".join(agent_outputs)
        prompt = (
            f"Synthesise the following agent results into a coherent, comprehensive answer.\n\n"
            f"Original task: {original_task}\n\nAgent results:\n{combined[:6000]}"
        )
        try:
            resp = await get_http_client().post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.warning("synthesis_failed", error=str(exc))
        return "\n\n".join(agent_outputs)
