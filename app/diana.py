from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, Trainer
from transformers import pipeline
from chromadb.config import Settings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
import time



def newPrompt(user_question: str):

    from constants import (
        CHROMA_SETTINGS,
        DOCUMENT_MAP,
        EMBEDDING_MODEL_NAME,
        INGEST_THREADS,
        PERSIST_DIRECTORY,
        MODEL_ID,
        SOURCE_DIRECTORY,
    )

    #model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    #EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)
    access_token = "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"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
        token=access_token)


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        token=access_token
    )

    ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    # Define the folder for storing database
    SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

    PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

    #PERSIST_DIRECTORY = f"/content/DB"
    CHROMA_SETTINGS = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )

    system_prompt = """Eres una asistente útil mujer llamado Dayana, utilizarás el contexto proporcionado para responder preguntas de los usuarios.
    Lee el contexto dado antes de responder preguntas y piensa paso a paso.
    Si no puedes responder una pregunta del usuario basándote en el contexto proporcionado,
    informa al usuario. No utilices ninguna otra información para responder al usuario. Proporciona una respuesta detallada a la pregunta. Responde siempre en español."""

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS

    instruction = """
    Context: {context}
    User: {question}"""

    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})
    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    pipe = pipeline("text-generation",
        model=model,
        tokenizer= tokenizer,
        #torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens = 4096,
        do_sample=True,
        #top_k=10,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        num_return_sequences=1,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id
    )


    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
        retriever=retriever,
        return_source_documents=True,
        #verbose=True,
        callbacks=callback_manager,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )

    #user_question = "¿Qué sabes del documento?"
    #start_time = 0
    #user_question = input("\nEnter a query: ")

    qa_chain_response = qa.stream(
        {"query": user_question},
    )

    return qa_chain_response["result"]
    #yield qa_chain_response
   