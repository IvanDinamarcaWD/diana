from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from diana import newPrompt
from typing import Dict, Iterator

#import time
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

        data_generator = newPrompt(question)
        #async def data_generator_func():
        #    for item in data_generator:
        #        print(item)
        #        # Yield each item serialized as JSON
        #        yield item['result']
                
        #def generate():
        #    for _ in range(5):  # Simulate 5 chunks of fake data
        #        time.sleep(1)  # Simulate some processing time
        #        yield f"Received message: {message}\n".encode("utf-8")
        return StreamingResponse(newPrompt(question), content_type="text/event-stream")
    
    except KeyError:
        raise HTTPException(status_code=400, detail="Message not provided in request.")
