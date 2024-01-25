from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from diana import newPrompt, create_gen
from typing import Dict, Iterator
import asyncio
import time
from langchain.callbacks import AsyncIteratorCallbackHandler

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/prompt")
def prompt(data: dict):
    try:
        question = data["message"]

        stream_it = AsyncIteratorCallbackHandler()
        gen = create_gen(question, stream_it)

        return StreamingResponse(gen, media_type="text/event-stream")
        #answer = newPrompt(question)
        #return StreamingResponse(_generate(answer), media_type="text/event-stream")

    except KeyError:
        raise HTTPException(status_code=400, detail="Message not provided in request.")
