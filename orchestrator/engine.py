"""Low-overhead asynchronous Gemini orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from pydantic import TypeAdapter

from schemas.workflow import OperationalTask, WorkflowState
from tools.operations import complete_task, record_extraction


class AgentFlowError(RuntimeError):
    """Base error for AgentFlow failures."""


class EmptyModelResponseError(AgentFlowError):
    """Raised when Gemini returns no usable candidate."""


ToolFunction = Callable[..., dict[str, Any]]


class AgentFlowEngine:
    """Run structured extraction and manual, locally executed tool calls."""

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str = "gemini-2.5-flash",
        max_turns: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self._tool_functions: dict[str, ToolFunction] = {
            "record_extraction": record_extraction,
            "complete_task": complete_task,
        }

    @staticmethod
    def _require_candidate(response: Any) -> Any:
        candidates = getattr(response, "candidates", None) or []
        if not candidates or getattr(candidates[0], "content", None) is None:
            raise EmptyModelResponseError("Gemini returned no usable candidate")
        return candidates[0]

    async def extract_structured_tasks(self, source_text: str) -> list[OperationalTask]:
        """Extract flat task entities with Gemini's native structured output."""
        prompt = "Extract operational tasks from this text. Return an empty list when none exist.\n" + source_text.strip()
        config = types.GenerateContentConfig(
            system_instruction="Extract tasks precisely. Do not infer missing facts.",
            response_mime_type="application/json",
            response_schema=list[OperationalTask],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        self._require_candidate(response)
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return TypeAdapter(list[OperationalTask]).validate_python(parsed)
        text = getattr(response, "text", None)
        if not text:
            raise EmptyModelResponseError("Gemini returned no structured extraction")
        return TypeAdapter(list[OperationalTask]).validate_json(text)

    async def _execute_tool_call(self, call: Any) -> dict[str, Any]:
        name = getattr(call, "name", None)
        function = self._tool_functions.get(name)
        if function is None:
            raise AgentFlowError(f"Unknown tool requested by Gemini: {name}")
        arguments = dict(getattr(call, "args", None) or {})
        return await asyncio.to_thread(function, **arguments)

    async def run_orchestration_loop(self, source_text: str) -> WorkflowState:
        """Extract tasks, then manually route and execute Gemini tool calls."""
        state = WorkflowState(source_text=source_text.strip(), status="running")
        state.tasks = await self.extract_structured_tasks(state.source_text)
        history: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self._routing_prompt(state))],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction="Route tasks to the available operations. Use tools only when needed.",
            tools=[record_extraction, complete_task],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for _ in range(self.max_turns):
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=history,
                config=config,
            )
            candidate = self._require_candidate(response)
            model_content = candidate.content
            history.append(model_content)
            function_calls = list(getattr(response, "function_calls", None) or [])
            if not function_calls:
                state.status = "completed"
                return state

            # One model turn can contain several calls. Keep all responses in one
            # user content so Gemini can pair them with the originating calls.
            results = await asyncio.gather(
                *(self._execute_tool_call(call) for call in function_calls)
            )
            response_parts: list[types.Part] = []
            for call, result in zip(function_calls, results, strict=True):
                state.tool_results.append(result)
                if result.get("completed") and result.get("title"):
                    state.completed_task_titles.append(str(result["title"]))
                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                    )
                )
            history.append(types.Content(role="user", parts=response_parts))

        state.status = "failed"
        raise AgentFlowError(f"Maximum orchestration turns exceeded: {self.max_turns}")

    @staticmethod
    def _routing_prompt(state: WorkflowState) -> str:
        task_lines = "\n".join(
            f"- {task.title} [{task.priority}]" for task in state.tasks
        ) or "- No tasks extracted"
        return f"Tasks extracted from the source:\n{task_lines}\nRoute them as appropriate."
