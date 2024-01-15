
import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.chains import VectorDBQA
#callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)
from prompt_template_utils import (system_prompt)
from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
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


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(
        ["llama", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)

def main(device_type, show_sources, use_history, model_type, save_qa):
    
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

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

    model_name_or_path = "TheBloke/dolphin-2.2.1-mistral-7B-GPTQ"

    llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    #return model, tokenizer

    #llm =  LlamaCpp(callback_manager=callback_manager,**kwargs)

    #llm.streaming = True


    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=model_type, history=use_history)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )
    

    #return StreamingHttpResponse(openai_response_generator(), content_type="text/event-stream")

    while True:
        query = input("\nEnter a query: ")

        print("\n\nEvaluating the prompt...\n\n");

        for text in qa.stream(query, stop=["Q:"]):
            print(text)

        #res = qa(query)

        #answer = res["result"]

        # Print the result
        #print("\n\n> Question:")
        #print(query)
        #print("\n> Answer:")
        #print(answer)

if __name__ == "__main__":
    main()
