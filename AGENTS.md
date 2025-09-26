# Repository Guidelines

## Project Structure & Module Organization

- `whisperlivekit/` holds the Python package: high-level orchestration in `core.py`, the FastAPI entrypoint in `basic_server.py`, real-time audio flow in `audio_processor.py`, plus focussed subpackages (`simul_whisper/` for streaming inference, `translation/` for NLLB, `diarization/` for speaker logic, and `web/` for the bundled UI assets).
- `chrome-extension/` contains the browser recorder that can stream audio into the backend; keep frontend snippets in sync with `whisperlivekit/web/` when you touch shared components.
- Docs and assets live at the repo root (`README.md`, `available_models.md`, `architecture.png`); update these when behavior or supported models change.

## Build, Test & Development Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[sentence]  # local editable install with optional translation deps
whisperlivekit-server --model base --language en  # run the FastAPI server locally
python sync_extension.py  # copy updated web assets into the Chrome extension
```

- Use `uvicorn whisperlivekit.basic_server:app --reload` while iterating on the API to pick up code changes without restarting.

## Coding Style & Naming Conventions

- Target Python ≥3.12 and follow PEP 8: four-space indentation, `snake_case` modules/functions, `CamelCase` classes, and descriptive but concise names (e.g., `warmup.py`, `silero_vad_iterator.py`).
- Prefer explicit type hints on all public methods and functions using built-in types where possible (e.g., `str`, `int`) and `typing.Annotated` for metadata like dependencies. Reuse the shared `logging` configuration instead of `print`.
- Keep streaming logic pure-Python and isolate framework bindings near the edges (FastAPI, WebSockets) to ease portability.

#### FastAPI Specific Practices

- Use async/await for path operations and dependencies where I/O bound.
- Manage application lifespan with `@asynccontextmanager` and `lifespan` parameter instead of `@app.on_event`.
- Define configuration with Pydantic `BaseSettings` in a separate `config.py` module and inject via dependencies.
- Reuse dependencies with `Annotated[Type, Depends(dep_func)]` to avoid duplication.
- Pin FastAPI and dependencies in `pyproject.toml` or `requirements.txt` using semantic versioning, e.g., `fastapi>=0.115.0,<0.116.0`.
- For larger apps, structure as: `app/main.py`, `app/dependencies.py`, `app/routers/` subpackage.

#### Kotlin (Android) Specific Practices

- Follow Kotlin naming conventions: UpperCamelCase for classes, lowerCamelCase for functions and properties, descriptive names.
- Leverage Kotlin idioms: Use extension functions for readability, function types over Java interfaces, and destructuring declarations.
- Ensure null safety: Prefer non-nullable types, use safe calls (?.) and Elvis operator (?:).
- Use coroutines for asynchronous operations: Employ suspend functions and structured concurrency.
- For UI, use Jetpack Compose if applicable; follow MVVM architecture with ViewModel and LiveData/Flow.
- Mark experimental APIs with @RequiresOptIn and use @OptIn to consume them.
- Avoid 'get' prefixes for functions; use direct naming or verbs like 'find', 'build'.

#### Swift (iOS) Specific Practices

- Follow Swift naming: UpperCamelCase for types, lowerCamelCase for functions/properties, no suffixes for argument labels.
- Use value types (structs/enums) over classes for data; prefer immutable where possible.
- Leverage concurrency: Use async/await for asynchronous code, actors for shared mutable state.
- For UI, use SwiftUI; structure with @State, @Binding, @Observable.
- Apply @available attributes to members, not extensions, for platform availability.
- Use type aliases for readability; handle options with OptionSet for bitwise operations.
- Mark synchronous APIs unavailable in async contexts with @\_unavailableFromAsync.

## Testing Guidelines

- No automated suite exists yet; when adding new behavior, supply focused `pytest` modules under a new `tests/` directory and gate on deterministic fixtures (short audio clips stored outside the package or generated on the fly). For async code, use `unittest.IsolatedAsyncioTestCase` or `pytest-asyncio`.
- Always perform a manual integration pass: launch `whisperlivekit-server`, open `http://localhost:8000`, and validate incremental transcripts, diarization toggles, and translation flags you touched. Use `TestClient` in `with` context for lifespan events.
- For mobile: Test Android with JUnit/Robolectric, iOS with XCTest; focus on UI and async flows.

## Commit & Pull Request Guidelines

- Recent history favors short, present-tense summaries (`demo extension`, `language detection after few seconds working`). Mirror that format and keep commits scoped.
- Each PR should describe the user-visible change, list CLI flags or models exercised, and attach screenshots or logs when UI or diarization output changes.
- Link related issues/discussions and note any model weight or optional dependency updates so reviewers can reproduce your environment.

## Security & Configuration Tips

- Store SSL material outside the repo; pass paths with `--ssl-certfile/--ssl-keyfile` when testing HTTPS.
- Large models are not vendored—document any manual downloads and avoid committing weights or cache directories.
