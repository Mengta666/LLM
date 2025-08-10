import torch

from datasets import load_dataset

dataset = load_dataset("csv", data_files = {
    "train": r".\datasets\subjQA\datasets_train.csv",
    "test": r".\datasets\subjQA\datasets_test.csv",
    "validation": r".\datasets\subjQA\datasets_validation.csv"
})

# print(dataset)

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(dataset["train"][13]["question"],dataset["train"][13]["context"], return_tensors='pt')
print(tokenizer.decode(inputs["input_ids"][0]))
# [CLS] how are the keys of the keyboard? [SEP] i was reluctant to try a wireless keyboard, but due to a wire - chomping kitty, decided it was best to go wireless. i ' m so glad i did. this keyboard is sleek and stylish. it has a great feel under my fingertips. i was concerned that a wireless keyboard would be & # 34 ; buggy & # 34 ; and not be efficient, but this keyboard is as good as any corded keyboard. it charges easily via usb port and holds a charge for about ten days. the illuminated keys are helpful, if, like me, your eyes aren ' t as young as they once were. i already had the logitech unifying plug that plugs into my computer for my mouse and touchpad. i turned the keyboard on and the logitech plug recognized it right away. i highly recommend this keyboard. answernotfound [SEP]
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
outputs = model(**inputs)
# 取最大的logits概率
start_logits = outputs.start_logits
end_logits = outputs.end_logits
import torch
start = torch.argmax(start_logits)
end = torch.argmax(end_logits)
print(start, end)
print(f"question: {dataset["train"][13]["question"]}")
print(tokenizer.decode(inputs["input_ids"][0][start:end+1]))
# tensor(16) tensor(116)
# question: How are the keys of the  keyboard?
# wireless keyboard, but due to a wire - chomping kitty, decided it was best to go wireless. i ' m so glad i did. this keyboard is sleek and stylish. it has a great feel under my fingertips. i was concerned that a wireless keyboard would be & # 34 ; buggy & # 34 ; and not be efficient, but this keyboard is as good as any corded keyboard. it charges easily via usb port and holds a charge for about ten days. the illuminated
