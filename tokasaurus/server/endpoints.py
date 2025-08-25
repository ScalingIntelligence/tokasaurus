import asyncio
import pickle
from contextlib import asynccontextmanager
import pickle
import time
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Path, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.exception_handlers import http_exception_handler
from openai.pagination import SyncPage
from openai.types.batch import Batch
from openai.types.file_deleted import FileDeleted
from openai.types.file_object import FileObject
from openai.types.model import Model
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from tokasaurus.common_types import (
    Engine,
    ServerConfig,
    TimedBarrier,
)
from tokasaurus.server.types import (
    BatchCompletionsRequest,
    BatchCreationRequest,
    BatchFileLine,
    ChatCompletionRequest,
    CompletionsRequest,
    FileEntry,
    SubmittedBatch,
    SubmittedBatchItem,
    CartridgeCompletionsRequest,
    CartridgeChatCompletionRequest,
    SynchronousBatchCartridgeChatCompletionsRequest,
    SynchronousBatchCompletionsRequest,
    nowstamp,
)
from tokasaurus.server.utils import (
    ServerState,
    generate_output,
    handle_batch,
    make_batch_status,
    process_chat_completions_output,
    process_completions_output,
    process_cartridge_chat_completions_output,
    process_request,
    receive_from_manager_loop,
    submit_request,
    with_cancellation,
)
from tokasaurus.utils import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    state_bundle: ServerState = app.state.state_bundle

    task = asyncio.create_task(receive_from_manager_loop(state_bundle))

    yield

    task.cancel()


app = FastAPI(lifespan=lifespan)


def create_streaming_response(completion_response, request):
    """
    Convert a completed ChatCompletion to a streaming response format.
    This waits for the entire completion and then streams it as SSE events.
    """
    import json

    
    async def generate_stream():
        # First chunk with role
        first_chunk = ChatCompletionChunk(
            id=completion_response.id,
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=""),
                    finish_reason=None
                )
            ],
            created=completion_response.created,
            model=completion_response.model,
            object="chat.completion.chunk"
        )
        yield f"data: {json.dumps(first_chunk.model_dump())}\n\n"
        
        # Content chunk with the full message
        for choice in completion_response.choices:
            content_chunk = ChatCompletionChunk(
                id=completion_response.id,
                choices=[
                    StreamChoice(
                        index=choice.index,
                        delta=ChoiceDelta(content=choice.message.content),
                        finish_reason=None
                    )
                ],
                created=completion_response.created,
                model=completion_response.model,
                object="chat.completion.chunk"
            )
            yield f"data: {json.dumps(content_chunk.model_dump())}\n\n"
        
        # Final chunk with finish reason
        for choice in completion_response.choices:
            final_chunk = ChatCompletionChunk(
                id=completion_response.id,
                choices=[
                    StreamChoice(
                        index=choice.index,
                        delta=ChoiceDelta(),
                        finish_reason=choice.finish_reason
                    )
                ],
                created=completion_response.created,
                model=completion_response.model,
                object="chat.completion.chunk"
            )
            yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
        
        # End of stream
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler_w_logging(request: Request, exc: HTTPException):
    state: ServerState = app.state.state_bundle
    state.logger.error(f"HTTPException: {exc.status_code} {exc.detail}")
    return await http_exception_handler(request, exc)

@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/v1/completions")
@with_cancellation
async def oai_completions(request: CompletionsRequest, raw_request: Request):
    state: ServerState = app.state.state_bundle
    req, out = await generate_output(state, request)
    return process_completions_output(state, request, req, out)


@app.post("/v1/chat/completions")
@with_cancellation
async def oai_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    state: ServerState = app.state.state_bundle
    req, out = await generate_output(state, request)
    output = process_chat_completions_output(state, request, req, out)

    if request.stream:
        return create_streaming_response(output, request)
    else:
        return output



@app.post("/v1/files", response_model=FileObject)
async def upload_file(
    file: UploadFile,
    purpose: str = Form(...),
) -> FileObject:
    state: ServerState = app.state.state_bundle

    content = await file.read()
    state.logger.debug(f"Received file: {file.filename}, size: {len(content)} bytes")

    # Create a file object response
    file_object = FileObject(
        id=str(uuid4()),
        bytes=len(content),
        created_at=nowstamp(),
        filename=file.filename,
        purpose=purpose,
        object="file",
        status="uploaded",
    )

    state.fid_to_file[file_object.id] = FileEntry(
        content=content,
        details=file_object,
    )

    return file_object


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(
    file_id: str = Path(..., description="The ID of the file to retrieve"),
):
    state: ServerState = app.state.state_bundle

    if file_id not in state.fid_to_file:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    content = state.fid_to_file[file_id].content

    return Response(content=content, media_type="application/octet-stream")


@app.delete("/v1/files/{file_id}")
async def delete_file(
    file_id: str = Path(..., description="The ID of the file to delete"),
):
    state: ServerState = app.state.state_bundle

    if file_id not in state.fid_to_file:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    del state.fid_to_file[file_id]

    return FileDeleted(id=file_id, deleted=True, object="file")


@app.post("/v1/batches", response_model=Batch)
async def create_batch(request: BatchCreationRequest):
    state: ServerState = app.state.state_bundle

    # Create a new batch ID
    batch_id = str(uuid4())

    fid = request.input_file_id
    if (file_entry := state.fid_to_file.get(fid)) is None:
        raise HTTPException(status_code=404, detail=f"File not found: {fid}")

    if file_entry.details.purpose != "batch":
        raise HTTPException(
            status_code=400,
            detail=f"File {fid} has purpose {file_entry.details.purpose}, not 'batch'",
        )

    # parse the file contents as JSONL
    file_content = file_entry.content.decode("utf-8")
    lines = file_content.splitlines()

    match request.endpoint:
        case "/v1/completions":
            request_type = CompletionsRequest
        case "/v1/chat/completions":
            request_type = ChatCompletionRequest
        case "/v1/cartridge/completions":
            request_type = CartridgeCompletionsRequest
        case "/v1/cartridge/chat/completions":
            request_type = CartridgeChatCompletionRequest
        case _:
            raise HTTPException(
                status_code=400, detail=f"Unsupported endpoint: {request.endpoint}"
            )

    parsed_lines = []
    for i, line in enumerate(lines):
        try:
            parsed = BatchFileLine.model_validate_json(line)
            assert parsed.url == request.endpoint, (
                f"Mismatch between line url of '{parsed.url}' and endpoint of "
                f"'{request.endpoint}'"
            )
            parsed_body = request_type.model_validate(parsed.body)
            parsed_lines.append((parsed, parsed_body))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Line {i} did not parse: {e}",
            )

    batch_items = []
    for parsed, parsed_body in parsed_lines:
        req = process_request(state, parsed_body)
        submitted = submit_request(state, req)
        batch_item = SubmittedBatchItem(
            line=parsed, user_req=parsed_body, submitted_req=submitted
        )
        batch_items.append(batch_item)

    handler_task = asyncio.create_task(handle_batch(state, batch_id))

    batch = SubmittedBatch(
        id=batch_id,
        creation_request=request,
        items=batch_items,
        task=handler_task,
    )
    state.bid_to_batch[batch_id] = batch

    return make_batch_status(batch)


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(
    batch_id: str = Path(..., description="The ID of the batch to retrieve"),
):
    state: ServerState = app.state.state_bundle

    if (batch := state.bid_to_batch.get(batch_id)) is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    return make_batch_status(batch)


@app.get("/v1/models")
async def list_models():
    state: ServerState = app.state.state_bundle

    return SyncPage(
        object="list",
        data=[
            Model(
                id=state.config.model,
                created=nowstamp(),
                object="model",
                owned_by="tokasaurus",
            ),
        ],
    )

### ------------------------------------------------------------
### BEGIN NON-OAI ENDPOINTS
### ------------------------------------------------------------

@app.post("/custom/synchronous-batch-completions")
@with_cancellation
async def synchronous_batch_completions(
    request: SynchronousBatchCompletionsRequest, raw_request: Request
):
    state: ServerState = app.state.state_bundle

    async def generate_and_process(req: ChatCompletionRequest):
        internal_req, output = await generate_output(state, req)
        return process_chat_completions_output(state, req, internal_req, output)

    # Create tasks for each request
    tasks = [asyncio.create_task(generate_and_process(req)) for req in request.requests]

    # Wait for all tasks to complete and collect results in order
    results = await asyncio.gather(*tasks)

    pickled_content = pickle.dumps(results)
    return Response(content=pickled_content, media_type="application/octet-stream")




@app.post("/custom/cartridge/completions")
@with_cancellation
async def cartridge_completions(request: CartridgeCompletionsRequest, raw_request: Request):
    state: ServerState = app.state.state_bundle
    req, out = await generate_output(state, request)
    return process_completions_output(state, request, req, out)


@app.post("/custom/cartridge/chat/completions")
@with_cancellation
async def cartridge_chat_completions(request: CartridgeChatCompletionRequest, raw_request: Request):
    state: ServerState = app.state.state_bundle
    req, out = await generate_output(state, request)
    return process_cartridge_chat_completions_output(state, request, req, out)


@app.post("/custom/cartridge/synchronous-batch-completions")
@with_cancellation
async def cartridge_synchronous_chat_completions(request: SynchronousBatchCartridgeChatCompletionsRequest, raw_request: Request):
    state: ServerState = app.state.state_bundle
    t0 = time.time()
    async def generate_and_process(req: CartridgeChatCompletionRequest):
        internal_req, output = await generate_output(state, req)
        return process_cartridge_chat_completions_output(state, req, internal_req, output)
    
    # Create tasks for each request
    tasks = [asyncio.create_task(generate_and_process(req)) for req in request.requests]
    
    # Wait for all tasks to complete and collect results in order
    results = await asyncio.gather(*tasks)
    t1 = time.time()
    print(f"synchronous_batch_completions took {t1 - t0} seconds")

    # return {"completions": results}

    pickled_content = pickle.dumps(results)
    return Response(
        content=pickled_content, media_type="application/octet-stream"
    )

### ------------------------------------------------------------
### END NON-OAI ENDPOINTS
### ------------------------------------------------------------



def start_server(
    config: ServerConfig,
    engines: list[Engine],
    process_name: str,
    barrier: TimedBarrier,
):
    setup_logging(config)

    state = ServerState(
        config=config,
        engines=engines,
        process_name=process_name,
    )
    state.logger.info("Starting web server")
    app.state.state_bundle = state

    barrier.wait()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
        log_level=config.uvicorn_log_level,
    )


if __name__ == "__main__":
    # Useful for debugging
    app.state.state_bundle = None

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10210,
        log_level="info",
    )
