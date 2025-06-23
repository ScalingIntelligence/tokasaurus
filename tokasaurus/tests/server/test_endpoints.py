import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Assuming your FastAPI app instance is named 'app' in 'tokasaurus.server.endpoints'
# Adjust the import path if your app instance is located elsewhere or named differently.
# We might need to initialize ServerState or mock it.
from tokasaurus.server.endpoints import app
from tokasaurus.server.types import ServerState, CompletionsRequest, ChatCompletionRequest
from tokasaurus.common_types import ServerConfig, Engine, Request, Response


@pytest.fixture
def client():
    # Mock expensive state initialization if necessary
    # For now, let's assume a minimal ServerState or mock its usage within endpoints
    mock_server_config = ServerConfig(model_path="test", tokenizer_path="test", port=1234)

    # If ServerState requires engines, mock them too
    mock_engine = AsyncMock(spec=Engine)

    # Simplified ServerState for testing. Adjust if more complex setup is needed.
    # The key is to mock what `generate_output` and other utilities expect.
    test_server_state = ServerState(
        config=mock_server_config,
        engines=[mock_engine], # Assuming at least one engine is expected
        process_name="test_server"
    )
    app.state.state_bundle = test_server_state
    return TestClient(app)

# --- Tests for /v1/completions ---

# Mocking the function that would actually process the request and generate output
# This allows us to test the validation and request handling part in isolation.
@patch("tokasaurus.server.endpoints.generate_output", new_callable=AsyncMock)
def test_completions_with_max_completion_tokens(mock_generate_output, client):
    # Mock the return value of generate_output
    # It should return a tuple (TokasaurusRequest, RequestOutput)
    # For simplicity, we're using AsyncMock for these complex objects for now.
    # A more accurate mock would involve creating instances of TokasaurusRequest and RequestOutput.
    mock_tok_request = AsyncMock()
    mock_req_output = AsyncMock()
    mock_generate_output.return_value = (mock_tok_request, mock_req_output)

    # Mock process_completions_output to prevent it from running its full logic
    with patch("tokasaurus.server.endpoints.process_completions_output") as mock_process_output:
        mock_process_output.return_value = {"id": "cmpl-test", "choices": [{"text": "test"}]} # Simplified OpenAI like response

        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Hello", "max_completion_tokens": 50},
        )
        assert response.status_code == 200
        # Check that generate_output was called, and its 'request' argument (CompletionsRequest)
        # has max_tokens correctly set.
        mock_generate_output.assert_called_once()
        called_args, _ = mock_generate_output.call_args
        assert len(called_args) > 1  # state, request_obj
        request_obj = called_args[1]
        assert isinstance(request_obj, CompletionsRequest)
        assert request_obj.max_tokens == 50
        assert request_obj.max_completion_tokens == 50 # The field is still there
        mock_process_output.assert_called_once()

@patch("tokasaurus.server.endpoints.generate_output", new_callable=AsyncMock)
def test_completions_with_max_tokens(mock_generate_output, client):
    mock_tok_request = AsyncMock()
    mock_req_output = AsyncMock()
    mock_generate_output.return_value = (mock_tok_request, mock_req_output)
    with patch("tokasaurus.server.endpoints.process_completions_output") as mock_process_output:
        mock_process_output.return_value = {"id": "cmpl-test", "choices": [{"text": "test"}]}

        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Hello", "max_tokens": 40},
        )
        assert response.status_code == 200
        mock_generate_output.assert_called_once()
        called_args, _ = mock_generate_output.call_args
        request_obj = called_args[1]
        assert isinstance(request_obj, CompletionsRequest)
        assert request_obj.max_tokens == 40
        assert request_obj.max_completion_tokens is None # Not provided
        mock_process_output.assert_called_once()

@patch("tokasaurus.server.endpoints.generate_output", new_callable=AsyncMock)
def test_completions_with_neither_token_param(mock_generate_output, client):
    mock_tok_request = AsyncMock()
    mock_req_output = AsyncMock()
    mock_generate_output.return_value = (mock_tok_request, mock_req_output)
    with patch("tokasaurus.server.endpoints.process_completions_output") as mock_process_output:
        mock_process_output.return_value = {"id": "cmpl-test", "choices": [{"text": "test"}]}

        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Hello"},
        )
        assert response.status_code == 200
        mock_generate_output.assert_called_once()
        called_args, _ = mock_generate_output.call_args
        request_obj = called_args[1]
        assert isinstance(request_obj, CompletionsRequest)
        assert request_obj.max_tokens == 16 # Default for CompletionsRequest
        assert request_obj.max_completion_tokens is None
        mock_process_output.assert_called_once()

# No need to patch generate_output here as validation happens before it's called
def test_completions_with_both_token_params(client):
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 30,
            "max_completion_tokens": 50,
        },
    )
    assert response.status_code == 422 # Unprocessable Entity due to Pydantic validation error
    assert "Only one of 'max_tokens' or 'max_completion_tokens' can be set." in response.json()["detail"][0]["msg"]

# --- Tests for /v1/chat/completions ---

@patch("tokasaurus.server.endpoints.generate_output", new_callable=AsyncMock)
def test_chat_completions_with_max_completion_tokens(mock_generate_output, client):
    mock_tok_request = AsyncMock()
    mock_req_output = AsyncMock()
    mock_generate_output.return_value = (mock_tok_request, mock_req_output)
    with patch("tokasaurus.server.endpoints.process_chat_completions_output") as mock_process_output:
        mock_process_output.return_value = {"id": "chatcmpl-test", "choices": [{"message": {"role": "assistant", "content": "Hi"}}]}

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 70,
            },
        )
        assert response.status_code == 200
        mock_generate_output.assert_called_once()
        called_args, _ = mock_generate_output.call_args
        request_obj = called_args[1]
        assert isinstance(request_obj, ChatCompletionRequest)
        assert request_obj.max_tokens == 70
        assert request_obj.max_completion_tokens == 70
        mock_process_output.assert_called_once()

@patch("tokasaurus.server.endpoints.generate_output", new_callable=AsyncMock)
def test_chat_completions_with_max_tokens(mock_generate_output, client):
    mock_tok_request = AsyncMock()
    mock_req_output = AsyncMock()
    mock_generate_output.return_value = (mock_tok_request, mock_req_output)
    with patch("tokasaurus.server.endpoints.process_chat_completions_output") as mock_process_output:
        mock_process_output.return_value = {"id": "chatcmpl-test", "choices": [{"message": {"role": "assistant", "content": "Hi"}}]}

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 60,
            },
        )
        assert response.status_code == 200
        mock_generate_output.assert_called_once()
        called_args, _ = mock_generate_output.call_args
        request_obj = called_args[1]
        assert isinstance(request_obj, ChatCompletionRequest)
        assert request_obj.max_tokens == 60
        assert request_obj.max_completion_tokens is None
        mock_process_output.assert_called_once()

@patch("tokasaurus.server.endpoints.generate_output", new_callable=AsyncMock)
def test_chat_completions_with_neither_token_param(mock_generate_output, client):
    mock_tok_request = AsyncMock()
    mock_req_output = AsyncMock()
    mock_generate_output.return_value = (mock_tok_request, mock_req_output)
    with patch("tokasaurus.server.endpoints.process_chat_completions_output") as mock_process_output:
        mock_process_output.return_value = {"id": "chatcmpl-test", "choices": [{"message": {"role": "assistant", "content": "Hi"}}]}

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        mock_generate_output.assert_called_once()
        called_args, _ = mock_generate_output.call_args
        request_obj = called_args[1]
        assert isinstance(request_obj, ChatCompletionRequest)
        assert request_obj.max_tokens is None # Default for ChatCompletionRequest max_tokens
        assert request_obj.max_completion_tokens is None
        mock_process_output.assert_called_once()

def test_chat_completions_with_both_token_params(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 55,
            "max_completion_tokens": 75,
        },
    )
    assert response.status_code == 422 # Unprocessable Entity
    assert "Only one of 'max_tokens' or 'max_completion_tokens' can be set for ChatCompletionRequest." in response.json()["detail"][0]["msg"]

# It might be good to have a conftest.py for the client fixture if more test files are added.
# For now, keeping it here is fine.
# Also, we need to ensure pytest and httpx are available in the environment.
# If not, I'll add a step to install them.
# The ServerState setup is minimal; if endpoints rely on more complex state,
# this fixture would need to be expanded or specific parts mocked more deeply.
# The mocks for tokasaurus.common_types.Request and Response might be too generic.
# If specific attributes of these are accessed, the mocks might need to be more detailed.
# For now, the primary check is that the `request_obj` (CompletionsRequest/ChatCompletionRequest)
# passed to `generate_output` has the correct `max_tokens` value after our Pydantic validation.
print("Initial test file created. Running these tests would require pytest and httpx.")
print("The tests mock generate_output and process_completions_output/process_chat_completions_output")
print("to focus on the validation logic within the Pydantic models when requests hit the endpoints.")

# Need to create __init__.py files for test directories to be importable modules
# tokasaurus/tests/__init__.py
# tokasaurus/tests/server/__init__.py
# I will do this in separate steps if this tool call succeeds.
# For now, I'll assume the user or CI environment handles pytest discovery.
# Or, I can add them now. Let's add them.

# This file assumes pytest can find and run these tests.
# The ServerState initialization in the fixture is a guess and might need refinement
# based on what `generate_output` actually needs from the state.
# The current mocking of `generate_output` and `process_*_output` is crucial
# as it isolates the endpoint's request validation and parsing logic.
# We are asserting that the `CompletionsRequest` or `ChatCompletionRequest` object,
# which is the first argument to `generate_output` after `state`, has the correct
# `max_tokens` value set by our Pydantic validators.
# The `max_completion_tokens` field itself will also be present on the model if it was in the input JSON,
# as Pydantic includes all fields defined on the model.
# Our validator just ensures `max_tokens` gets the right value.
# The assertion `assert request_obj.max_completion_tokens == 50` (for example) is correct
# because the field is part of the model definition.
# If `max_completion_tokens` was *not* part of the model fields but just a temp var in validator,
# it wouldn't be on `request_obj`. But we did add it as `Optional[int] = None`.
# This seems fine.
# The error message check for the "both params" case is specific to Pydantic v2's error structure
# for @model_validator(mode='before') which wraps the error in a list under 'detail'.
# If using Pydantic v1, root_validator errors might format differently. Assuming v2 based on `model_validator`.
# The common_types.Request and Response are not used in these specific mocks currently.
# The ServerConfig paths are dummy values, as they are not directly used by these tests due to mocking.
# The mock_engine for ServerState is also basic; if generate_output interacts with it, it might need more setup.
# The key is that FastAPI constructs the Request object (CompletionsRequest/ChatCompletionRequest)
# and our validators run. We then check that this object, when passed to the next function
# (generate_output), has the correct `max_tokens` value.
# The `process_*_output` functions are also mocked to prevent them from executing their full logic,
# as we only care that they are called after `generate_output`.
# The structure with `app.state.state_bundle = test_server_state` is a common way to set up
# state for `TestClient` if the app uses `request.app.state`. Here, it seems endpoints directly use `app.state`.
# This should work.

# One final thought: the import `from tokasaurus.common_types import Request, Response` might be problematic
# if these are very generic names and conflict. However, within the current scope, they are not used.
# The important ones are `ServerState`, `CompletionsRequest`, `ChatCompletionRequest`, `ServerConfig`, `Engine`.
# `CompletionsRequest` and `ChatCompletionRequest` are correctly imported from `tokasaurus.server.types`.
# `ServerState` is also from `tokasaurus.server.types`.
# `ServerConfig` and `Engine` are from `tokasaurus.common_types`. This seems fine.
# The test file structure:
# tokasaurus/
#   server/
#     endpoints.py
#     types.py
#   tests/
#     server/
#       test_endpoints.py
# This is a standard layout.
# I will create the __init__.py files next.```text
Initial test file created. Running these tests would require pytest and httpx.
The tests mock generate_output and process_completions_output/process_chat_completions_output
to focus on the validation logic within the Pydantic models when requests hit the endpoints.
Need to create __init__.py files for test directories to be importable modules
tokasaurus/tests/__init__.py
tokasaurus/tests/server/__init__.py
I will do this in separate steps if this tool call succeeds.
For now, I'll assume the user or CI environment handles pytest discovery.
Or, I can add them now. Let's add them.
This file assumes pytest can find and run these tests.
The ServerState initialization in the fixture is a guess and might need refinement
based on what `generate_output` actually needs from the state.
The current mocking of `generate_output` and `process_*_output` is crucial
as it isolates the endpoint's request validation and parsing logic.
We are asserting that the `CompletionsRequest` or `ChatCompletionRequest` object,
which is the first argument to `generate_output` after `state`, has the correct
`max_tokens` value set by our Pydantic validators.
The `max_completion_tokens` field itself will also be present on the model if it was in the input JSON,
as Pydantic includes all fields defined on the model.
Our validator just ensures `max_tokens` gets the right value.
The assertion `assert request_obj.max_completion_tokens == 50` (for example) is correct
because the field is part of the model definition.
If `max_completion_tokens` was *not* part of the model fields but just a temp var in validator,
it wouldn't be on `request_obj`. But we did add it as `Optional[int] = None`.
This seems fine.
The error message check for the "both params" case is specific to Pydantic v2's error structure
for @model_validator(mode='before') which wraps the error in a list under 'detail'.
If using Pydantic v1, root_validator errors might format differently. Assuming v2 based on `model_validator`.
The common_types.Request and Response are not used in these specific mocks currently.
The ServerConfig paths are dummy values, as they are not directly used by these tests due to mocking.
The mock_engine for ServerState is also basic; if generate_output interacts with it, it might need more setup.
The key is that FastAPI constructs the Request object (CompletionsRequest/ChatCompletionRequest)
and our validators run. We then check that this object, when passed to the next function
(generate_output), has the correct `max_tokens` value.
The `process_*_output` functions are also mocked to prevent them from executing their full logic,
as we only care that they are called after `generate_output`.
The structure with `app.state.state_bundle = test_server_state` is a common way to set up
state for `TestClient` if the app uses `request.app.state`. Here, it seems endpoints directly use `app.state`.
This should work.
One final thought: the import `from tokasaurus.common_types import Request, Response` might be problematic
if these are very generic names and conflict. However, within the current scope, they are not used.
The important ones are `ServerState`, `CompletionsRequest`, `ChatCompletionRequest`, `ServerConfig`, `Engine`.
`CompletionsRequest` and `ChatCompletionRequest` are correctly imported from `tokasaurus.server.types`.
`ServerState` is also from `tokasaurus.server.types`.
`ServerConfig` and `Engine` are from `tokasaurus.common_types`. This seems fine.
The test file structure:
tokasaurus/
  server/
    endpoints.py
    types.py
  tests/
    server/
      test_endpoints.py
This is a standard layout.
I will create the __init__.py files next.
```
