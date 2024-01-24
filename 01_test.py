from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
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

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Using the text streamer to stream output one token at a time
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

pipe = pipeline("text-generation",
    model=model,
    tokenizer= tokenizer,
    #torch_dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens = 512,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id
)

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS
)

retriever = db.as_retriever()
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

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

    query = input("\nEnter a query: ")
    start_time = time.time()

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
    print(elapsed_time)


