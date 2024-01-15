from fastapi import FastAPI, Response
import time
from fastapi.responses import StreamingResponse

app = FastAPI()

# Replace 'your_llama_cpp_method' with the actual method from langchain.llms.LlamaCpp
def your_llama_cpp_method():
    # Your logic here
    # For example, simulate streaming data
    for i in range(10):
        yield f"Data point {i}\n"
        time.sleep(1)


@app.post("/stream")
def stream_response(prompt : str):
    # Replace 'your_llama_cpp_method' with the actual method from langchain.llms.LlamaCpp
    data_generator = your_llama_cpp_method()
    
    return StreamingResponse(data_generator, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
