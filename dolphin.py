from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#model_name_or_path = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
model_name_or_path = "TheBloke/dolphin-2.2.1-mistral-7B-GPTQ"

# To use a different branch, change revision
# For example: revision="gptq-4bit-128g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

#prompt = "Write a story about llamas"
prompt = input("\nEnter a query: ")

system_message = "You are a story writing assistant"
prompt_template=f'''[INST] {prompt} [/INST]
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

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
