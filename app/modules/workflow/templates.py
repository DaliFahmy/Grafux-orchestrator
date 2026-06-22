from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.persistence.models import WorkflowTemplate
from app.modules.workflow.schemas import WorkflowTemplateResponse

# Default templates loaded on first request
_DEFAULT_TEMPLATES = [
    {
        "name": "Agentic Research",
        "category": "research",
        "description": "Multi-step internet research with summarization",
        "graph_definition": {
            "entry_node": "agent",
            "nodes": [
                {"name": "agent", "type": "agent", "config": {}},
                {"name": "research", "type": "research", "config": {}},
            ],
            "edges": [
                {"from_node": "agent", "to_node": "research", "condition": "needs_research"},
                {"from_node": "research", "to_node": "agent"},
            ],
        },
    },
    {
        "name": "Code Generation & Execution",
        "category": "sandbox",
        "description": "Generate Python code with an LLM and run it in E2B",
        "graph_definition": {
            "entry_node": "agent",
            "nodes": [
                {"name": "agent", "type": "agent", "config": {}},
                {"name": "sandbox", "type": "sandbox", "config": {}},
            ],
            "edges": [
                {"from_node": "agent", "to_node": "sandbox", "condition": "has_code"},
                {"from_node": "sandbox", "to_node": "agent"},
            ],
        },
    },
    {
        "name": "MCP Tool Orchestration",
        "category": "mcp",
        "description": "Agent that coordinates multiple MCP tools",
        "graph_definition": {
            "entry_node": "agent",
            "nodes": [
                {"name": "agent", "type": "agent", "config": {}},
                {"name": "mcp", "type": "mcp", "config": {}},
            ],
            "edges": [
                {"from_node": "agent", "to_node": "mcp", "condition": "has_tool_calls"},
                {"from_node": "mcp", "to_node": "agent"},
            ],
        },
    },
    {
        "name": "Device + AI Pipeline",
        "category": "devices",
        "description": "AI-directed robotic / device automation workflow",
        "graph_definition": {
            "entry_node": "agent",
            "nodes": [
                {"name": "agent", "type": "agent", "config": {}},
                {"name": "device", "type": "device", "config": {}},
            ],
            "edges": [
                {"from_node": "agent", "to_node": "device", "condition": "has_device_command"},
                {"from_node": "device", "to_node": "agent"},
            ],
        },
    },
]


class TemplateService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def list_templates(
        self, category: str | None = None
    ) -> list[WorkflowTemplateResponse]:
        q = select(WorkflowTemplate).where(WorkflowTemplate.is_public == True)  # noqa: E712
        if category:
            q = q.where(WorkflowTemplate.category == category)
        result = await self._db.execute(q.order_by(WorkflowTemplate.name))
        templates = list(result.scalars().all())
        return [WorkflowTemplateResponse.model_validate(t) for t in templates]

    async def seed_defaults(self) -> None:
        """Insert default templates if the table is empty."""
        result = await self._db.execute(select(WorkflowTemplate).limit(1))
        if result.scalar_one_or_none():
            return  # already seeded
        for td in _DEFAULT_TEMPLATES:
            self._db.add(WorkflowTemplate(**td))
        await self._db.commit()
