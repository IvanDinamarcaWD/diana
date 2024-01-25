from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from diana import newPrompt
from typing import Dict, Iterator

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

@app.post("/prompt")
def prompt(data: dict):
    try:
        question = data["message"]

        async def data_generator_func():
            data_generator = newPrompt(question)
            for item in data_generator:
                
                print(next(item))
                # Yield each item serialized as JSON
                #yield next(item['result'])

        data_generator_func()
        def generate():
            for _ in range(5):  # Simulate 5 chunks of fake data
                time.sleep(1)  # Simulate some processing time
                yield f"Received message: {question}\n".encode("utf-8")
        return StreamingResponse(generate())
    
    except KeyError:
        raise HTTPException(status_code=400, detail="Message not provided in request.")
