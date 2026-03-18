from transformers import AutoModelForCausalLM, AutoTokenizer
model=AutoModelForCausalLM.from_pretrained(
   "microsoft/Phi-3-mini-4k-instruct",
   torch_dtype="auto",
   trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
from transformers import pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,return_full_text=False,max_new_tokens=500,do_sample=False)
messages = [{"role":"user","content": "Create a funny joke about chickens"}]
output = pipe(messages, use_cache=False)
print(output[0]["generated_text"])
