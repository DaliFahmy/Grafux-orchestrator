from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_workflow_engine_build_graph():
    """WorkflowEngine.build_graph returns a compiled LangGraph graph."""
    from app.modules.workflow.engine import WorkflowEngine

    engine = WorkflowEngine()
    graph = engine.build_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_workflow_engine_validate_graph_valid():
    from app.modules.workflow.engine import WorkflowEngine

    engine = WorkflowEngine()
    valid_def = {
        "entry_node": "agent",
        "nodes": [{"name": "agent", "type": "agent", "config": {}}],
        "edges": [],
    }
    await engine.validate_graph(valid_def)  # should not raise


@pytest.mark.asyncio
async def test_workflow_engine_validate_graph_invalid():
    from app.modules.workflow.engine import WorkflowEngine

    engine = WorkflowEngine()
    with pytest.raises(ValueError, match="entry_node"):
        await engine.validate_graph({"nodes": []})


@pytest.mark.asyncio
async def test_execution_repository_create(db_session):
    from app.modules.persistence.repositories import ExecutionRepository

    repo = ExecutionRepository(db_session)
    execution = await repo.create(
        org_id="org-1",
        project_id="proj-1",
        triggered_by="user-1",
        input={"message": "hello"},
    )
    assert execution.id is not None
    assert execution.status == "pending"


@pytest.mark.asyncio
async def test_execution_repository_update_status(db_session):
    from datetime import datetime
    from app.modules.persistence.repositories import ExecutionRepository

    repo = ExecutionRepository(db_session)
    execution = await repo.create(org_id="org-1", project_id="proj-1", input={})
    await repo.update_status(execution.id, "running", started_at=datetime.utcnow())
    await db_session.commit()

    updated = await repo.get(execution.id)
    assert updated.status == "running"


@pytest.mark.asyncio
async def test_execution_not_found(db_session):
    from app.core.exceptions import ExecutionNotFoundError
    from app.modules.persistence.repositories import ExecutionRepository

    repo = ExecutionRepository(db_session)
    with pytest.raises(ExecutionNotFoundError):
        await repo.get_or_raise("nonexistent-id")


@pytest.mark.asyncio
async def test_event_bus_emit():
    """Event bus publishes to correct Redis channel."""
    with patch("app.modules.streaming.event_bus.publish", new_callable=AsyncMock) as mock_publish:
        from app.modules.streaming.event_bus import emit
        await emit("exec-123", "execution.started", {"org_id": "org-1"})
        mock_publish.assert_called_once()
        channel, message = mock_publish.call_args.args
        assert channel == "execution:exec-123"
        assert "execution.started" in message


@pytest.mark.asyncio
async def test_research_pipeline_no_api_key():
    """Research pipeline gracefully handles missing Tavily key."""
    with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
        from importlib import reload
        import app.config as cfg
        cfg.get_settings.cache_clear()

        from app.modules.research.pipeline import ResearchPipeline
        pipeline = ResearchPipeline()
        result = await pipeline.run("test query")
        assert "summary" in result
        assert result["sources"] == []
