from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline
from prompt_template_utils import get_prompt_template
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS
)
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

# Using the text streamer to stream output one token at a time
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)


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

generation_params = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1
    }
    
pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_params
    )

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


while True:
        
    #prompt =  "Tell me about AI"
    prompt = input("\nEnter a query: ")

    # Convert prompt to tokens
    tokens = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.cuda()

    

    # Generate streamed output, visible one token at a time
    generation_output = model.generate(
        tokens,
        streamer=streamer,
        **generation_params
    )

    # Generation without a streamer, which will include the prompt in the output
    generation_output = model.generate(
        tokens,
        **generation_params
    )

    # Get the tokens from the output, decode them, print them
    token_output = generation_output[0]
    text_output = tokenizer.decode(token_output)
    print("model.generate output: ", text_output)

    # Inference is also possible via Transformers' pipeline

    

    pipe_output = pipe(prompt_template)[0]['generated_text']
    print("pipeline output: ", pipe_output)
