from fastapi import FastAPI, HTTPException,streaming
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

def generate_data() -> Iterator[Dict[str, any]]:
    # Your function to generate data
    # This could be any function that yields dictionaries
    for i in range(10):
        yield {"index": i, "data": f"Data {i}"}

@app.post("/prompt")
def prompt(data: dict):
    try:
        question = data["message"]
        data_generator = newPrompt(question)

        async def data_generator_func():
            for item in data_generator:
                # Yield each item serialized as JSON
                yield streaming.StringResponse(content=item)
                
        #def generate():
        #    for _ in range(5):  # Simulate 5 chunks of fake data
        #        time.sleep(1)  # Simulate some processing time
        #        yield f"Received message: {message}\n".encode("utf-8")
        return StreamingResponse(data_generator_func())
    
    except KeyError:
        raise HTTPException(status_code=400, detail="Message not provided in request.")
