from __future__ import annotations

from types import SimpleNamespace
from typing import get_origin

import pytest
from google.genai import types
from pydantic import ValidationError

from orchestrator.engine import AgentFlowEngine


def model_response(*, parsed=None, calls=None, text="done"):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        parsed=parsed,
        text=text,
        function_calls=calls or [],
    )


class FakeGenerateContent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def __call__(self, *, model, contents, config):
        request_contents = list(contents) if isinstance(contents, list) else contents
        self.calls.append((model, request_contents, config))
        return next(self.responses)


class FakeClient:
    def __init__(self, responses):
        generate_content = FakeGenerateContent(responses)
        self.generate_content = generate_content
        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )


@pytest.mark.asyncio
async def test_extract_structured_tasks_uses_flat_schema_and_zero_thinking():
    client = FakeClient(
        [
            model_response(
                parsed=[
                    {
                        "title": "Send report",
                        "description": "Email the weekly report",
                        "priority": "high",
                    }
                ]
            )
        ]
    )
    engine = AgentFlowEngine(client)

    tasks = await engine.extract_structured_tasks("Send the weekly report.")

    assert tasks[0].title == "Send report"
    config = client.generate_content.calls[0][2]
    assert get_origin(config.response_schema) is list
    assert config.thinking_config.thinking_budget == 0


@pytest.mark.asyncio
async def test_invalid_structured_payload_raises_pydantic_validation_error():
    client = FakeClient(
        [model_response(parsed=[{"title": "", "priority": "urgent"}])]
    )

    with pytest.raises(ValidationError):
        await AgentFlowEngine(client).extract_structured_tasks("Bad input")


@pytest.mark.asyncio
async def test_single_tool_call_is_executed_manually():
    call = types.FunctionCall(
        name="complete_task",
        args={"title": "Ship release", "owner": "Ada"},
    )
    client = FakeClient(
        [
            model_response(parsed=[]),
            model_response(calls=[call]),
            model_response(),
        ]
    )

    state = await AgentFlowEngine(client).run_orchestration_loop("Ship release")

    assert state.status == "completed"
    assert state.completed_task_titles == ["Ship release"]
    routing_config = client.generate_content.calls[1][2]
    assert routing_config.automatic_function_calling.disable is True
    assert routing_config.thinking_config.thinking_budget == 0


@pytest.mark.asyncio
async def test_multiple_function_calls_are_batched_in_one_user_content_turn():
    calls = [
        types.FunctionCall(name="record_extraction", args={"title": "One"}),
        types.FunctionCall(name="complete_task", args={"title": "Two"}),
    ]
    client = FakeClient(
        [
            model_response(parsed=[]),
            model_response(calls=calls),
            model_response(),
        ]
    )

    state = await AgentFlowEngine(client).run_orchestration_loop("One and Two")

    assert state.status == "completed"
    second_routing_contents = client.generate_content.calls[2][1]
    response_turn = second_routing_contents[-1]
    assert response_turn.role == "user"
    assert len(response_turn.parts) == 2
    assert [part.function_response.name for part in response_turn.parts] == [
        "record_extraction",
        "complete_task",
    ]
    assert len([content for content in second_routing_contents if content.role == "user"]) == 2
