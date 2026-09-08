# AgentFlow

AgentFlow is a Streamlit app for structured task extraction and tool orchestration with Gemini.

## Live App

[Open AgentFlow](https://agentflow-11.streamlit.app/)

## Features

- Extract operational tasks from unstructured text.
- Run prompts through the AgentFlow orchestration engine.
- Inspect extracted tasks, tool calls, and execution results.

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```powershell
   streamlit run app.py
   ```

4. Enter a Gemini API key in the app sidebar.

## Tests

```powershell
pytest
```
