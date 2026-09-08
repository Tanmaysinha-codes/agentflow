from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import streamlit as st
from google import genai

from orchestrator.engine import AgentFlowEngine


MODEL_NAME = "gemini-2.5-flash"
T = TypeVar("T")


def run_async(operation: Callable[[], Awaitable[T]]) -> T:
    """Run an async engine operation from Streamlit, including nested-loop hosts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(operation())).result()


def build_engine(api_key: str) -> AgentFlowEngine:
    return AgentFlowEngine(genai.Client(api_key=api_key), model=MODEL_NAME)


def render_status(api_key: str | None) -> None:
    st.sidebar.subheader("Telemetry")
    st.sidebar.caption(f"Model: `{MODEL_NAME}`")
    st.sidebar.caption("Thinking budget: `0` (disabled)")
    st.sidebar.caption("Tool loop: `manual`")
    if api_key:
        st.sidebar.success("Gemini API key ready")
    else:
        st.sidebar.warning("Gemini API key required")


def render_extraction() -> None:
    st.subheader("Turn unstructured text into an operational task")
    source_text = st.text_area(
        "Raw input",
        value="Priority 4: Sync user record 88219 into staging table",
        height=150,
        key="extraction_input",
    )
    if st.button("Extract Task", type="primary", key="extract_task"):
        api_key = st.session_state.get("gemini_api_key", "").strip()
        if not api_key:
            st.warning("Enter a Gemini API key in the sidebar before extracting a task.")
            return
        if not source_text.strip():
            st.warning("Enter some raw input to extract.")
            return
        try:
            tasks = run_async(lambda: build_engine(api_key).extract_structured_tasks(source_text))
            st.session_state["extracted_tasks"] = [task.model_dump() for task in tasks]
        except Exception as error:
            st.error(f"Task extraction failed: {error}")

    if "extracted_tasks" in st.session_state:
        st.json(st.session_state["extracted_tasks"])


def render_trace(state: object) -> None:
    tasks = getattr(state, "tasks", [])
    results = getattr(state, "tool_results", [])
    with st.status("Execution trace", state="complete", expanded=True):
        if tasks:
            st.code("Extracted tasks:\n" + "\n".join(f"- {task.title} [{task.priority}]" for task in tasks))
        if results:
            for index, result in enumerate(results, start=1):
                st.code(f"Step {index}: tool result\n{result}", language="json")
        else:
            st.code("No tool calls were requested by the model.")


def render_orchestration() -> None:
    st.subheader("Route a high-level prompt through AgentFlow")
    prompt = st.text_input(
        "User prompt",
        value="Send a system notification to devops@company.local stating DB sync complete",
        key="orchestration_input",
    )
    if st.button("Run Orchestration Engine", type="primary", key="run_engine"):
        api_key = st.session_state.get("gemini_api_key", "").strip()
        if not api_key:
            st.warning("Enter a Gemini API key in the sidebar before running the engine.")
            return
        if not prompt.strip():
            st.warning("Enter a prompt to orchestrate.")
            return
        try:
            state = run_async(lambda: build_engine(api_key).run_orchestration_loop(prompt))
            st.session_state["workflow_state"] = state
        except Exception as error:
            st.session_state.pop("workflow_state", None)
            st.error(f"Orchestration failed: {error}")

    state = st.session_state.get("workflow_state")
    if state is not None:
        render_trace(state)
        if getattr(state, "status", None) == "completed":
            st.success("Workflow completed successfully.")
        else:
            st.error(f"Workflow ended with status: {getattr(state, 'status', 'unknown')}")


def main() -> None:
    st.set_page_config(page_title="AgentFlow", page_icon="⚡", layout="wide")
    st.title("AgentFlow")
    st.caption("An ultra-low-latency agentic engine for structured extraction and tool orchestration.")

    environment_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if environment_key:
        st.session_state["gemini_api_key"] = environment_key
    else:
        st.sidebar.text_input(
            "Gemini API key",
            type="password",
            key="gemini_api_key",
            help="Used only for this Streamlit session.",
        )
    api_key = st.session_state.get("gemini_api_key", "").strip()
    render_status(api_key or None)

    st.divider()
    extraction_tab, orchestration_tab = st.tabs(
        ["Structured Task Extraction", "Interactive Tool Orchestration"]
    )
    with extraction_tab:
        render_extraction()
    with orchestration_tab:
        render_orchestration()


if __name__ == "__main__":
    main()