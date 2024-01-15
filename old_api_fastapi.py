import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from prompt_template_utils import get_prompt_template
from huggingface_hub import hf_hub_download

from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)


from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
    CONTEXT_WINDOW_SIZE,
    N_BATCH
)
app = FastAPI()

async def generator(question):


    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "mps"})

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_BASENAME,
        resume_download=True,
        cache_dir=MODELS_PATH,
    )
    prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

    kwargs = {
        "model_path": model_path,
        "n_ctx": CONTEXT_WINDOW_SIZE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_batch": N_BATCH,  
        "n_gpu_layers": 1,
        "stream"  : True,
        "streaming"  : True
        #"verbose": True
    }

    llm =  LlamaCpp(callback_manager=callback_manager,**kwargs)

    llm.streaming = True


    chain = ConversationalRetrievalChain.from_llm(
            llm= llm,
            retriever= retriever,
            memory=memory,
            chain_type='stuff',
            combine_docs_chain_kwargs = { "prompt":prompt}    
        )


    run = asyncio.create_task(chain.arun(question))
    
    async for token in callback_manager.g
        yield token

    await run
    

@app.get('/', status_code=200)
async def get_test():
    return 'hola'

# Conversation Route
@app.post('/conversation',status_code=200)
async def get_conversation():
    return StreamingResponse(generator('hola'), media_type="text/event-stream")