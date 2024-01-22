from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from prompt_template_utils import get_prompt_template
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from prompt_template_utils import get_prompt_template
import time

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})
# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS
)
retriever = db.as_retriever()

# get the prompt template and memory if set by the user.
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

#
llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
    retriever=retriever,
    return_source_documents=True,  
    #verbose=True,
    #streaming: True,
    callbacks=callback_manager,
    chain_type_kwargs={
        "prompt": prompt,
    },
)
start_time = 0

while True:
    start_time = time.time()

    query = input("\nEnter a query: ")

    #for text in qa.stream(query, stop=["Q:"]):
    #        print(text)

    #
    if query == "exit":
        break
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Tiempo: {elapsed_time} segundos")


    
