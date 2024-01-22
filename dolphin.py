from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")


system_prompt = """Eres un asistente útil llamado Dayana, utilizarás el contexto proporcionado para responder preguntas de los usuarios. 
Lee el contexto dado antes de responder preguntas y piensa paso a paso. 
Si no puedes responder una pregunta del usuario basándote en el contexto proporcionado, 
informa al usuario. No utilices ninguna otra información para responder al usuario. Proporciona una respuesta detallada a la pregunta. Responde siempre en español."""


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = input("\nEnter a query: ")

prompt_template=f'''<s>[INST]{system_prompt} {prompt} [/INST]
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
#print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
