from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from app.modules.persistence.repositories import (
    EventRepository,
    ExecutionRepository,
    LogRepository,
    StepRepository,
)


@pytest.mark.asyncio
async def test_step_list_limit_and_offset(db_session):
    ex = await ExecutionRepository(db_session).create(
        org_id="o", project_id="p", input={}
    )
    steps = StepRepository(db_session)
    base = datetime(2024, 1, 1)
    for i in range(5):
        await steps.create(
            execution_id=ex.id, node_name=f"n{i}", started_at=base + timedelta(seconds=i)
        )
    await db_session.commit()

    first_two = await steps.list_by_execution(ex.id, limit=2, offset=0)
    assert [s.node_name for s in first_two] == ["n0", "n1"]

    next_two = await steps.list_by_execution(ex.id, limit=2, offset=2)
    assert [s.node_name for s in next_two] == ["n2", "n3"]

    # Default bounds are generous enough to return everything for a normal run.
    assert len(await steps.list_by_execution(ex.id)) == 5


@pytest.mark.asyncio
async def test_log_list_limit_and_offset(db_session):
    ex = await ExecutionRepository(db_session).create(
        org_id="o", project_id="p", input={}
    )
    logs = LogRepository(db_session)
    for i in range(5):
        await logs.append(ex.id, message=f"m{i}")
    await db_session.commit()

    assert len(await logs.list_by_execution(ex.id, limit=2)) == 2
    assert len(await logs.list_by_execution(ex.id, limit=10, offset=4)) == 1


@pytest.mark.asyncio
async def test_event_list_limit_and_offset(db_session):
    ex = await ExecutionRepository(db_session).create(
        org_id="o", project_id="p", input={}
    )
    events = EventRepository(db_session)
    for i in range(5):
        await events.append(ex.id, event_type="step", payload={"i": i})
    await db_session.commit()

    assert len(await events.list_by_execution(ex.id, limit=3)) == 3
    assert len(await events.list_by_execution(ex.id, limit=10, offset=4)) == 1
    # Previously unbounded — default now caps the result set.
    assert len(await events.list_by_execution(ex.id)) == 5
