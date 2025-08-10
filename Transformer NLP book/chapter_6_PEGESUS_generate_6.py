from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = "E:/python/LLM/Transformer NLP book/chapter_6/pegasus_summarizer"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

from datasets import load_dataset
dataset = load_dataset("knkarthick/samsum")
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
sample_text = dataset["test"][0]["dialogue"]
reference = dataset["test"][0]["summary"]
outputs = model.generate(**tokenizer(sample_text, return_tensors="pt"), **gen_kwargs)
print("Generate text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("Reference text:")
print(reference)
# Generate text:
# Amanda can't find Betty's number. Larry called her last time they were at the park together. Hannah doesn't know him well. Hannah will text Larry.
# Reference text:
# Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry.