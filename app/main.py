from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from diana import newPrompt
from typing import Dict, Iterator
import asyncio
import time

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _generate(answer: str):
	chunk_size = 10
	num_chunks = (len(answer) + chunk_size - 1) // chunk_size
	for i in range(num_chunks):
		chunk_start = i * chunk_size
		chunk_end = (i + 1) * chunk_size
		chunk = answer[chunk_start:chunk_end]
		response_body = f"{chunk}"
		yield response_body.encode()
		await asyncio.sleep(0.1)



@app.post("/prompt")
def prompt(data: dict):
    try:
        question = data["message"]
        answer = newPrompt(question)
        return StreamingResponse(_generate(answer), media_type="text/event-stream")

    except KeyError:
        raise HTTPException(status_code=400, detail="Message not provided in request.")
