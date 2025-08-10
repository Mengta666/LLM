from datasets import load_dataset
dataset = load_dataset("knkarthick/samsum")
print(dataset["train"].features)

from transformers import pipeline
model = pipeline("summarization", model="google/pegasus-cnn_dailymail")
print("Origin:")
print(dataset["train"][1]["summary"])
# Origin:
# Olivia and Olivier are voting for liberals in this election.
output = model(dataset["train"][1]["dialogue"])
print("Summary:")
summary = output[0]["summary_text"].replace("<n>", "\n")
print(summary)
# Summary:
# Oliver: Liberals as always. Olivia: Me too!!
# Oliver: Great.
# Olivia: Who's you voting for in this election?
# Oliver: Liberals as always.

# 使用ROUGE对PEGASUS在这个数据集上进行评估
import evaluate
rouge = evaluate.load("rouge")
sorce = rouge.compute(predictions=[summary], references=[dataset["train"][1]["summary"]])
import pandas as pd
print(pd.DataFrame.from_dict(sorce, orient="index", columns=["pegasus"]).T)
#          rouge1  rouge2  rougeL  rougeLsum
# pegasus  0.4375     0.2   0.375     0.4375

# 查看数据集的长度
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
dialog_len = [len(tokenizer.encode(s)) for s in dataset["train"]["dialogue"] if s is not None]
summary_len = [len(tokenizer.encode(s)) for s in dataset["train"]["summary"] if s is not None]
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].hist(dialog_len, bins=100, color="black", edgecolor="black")
axes[0].set_xlabel("Length")
axes[0].set_ylabel("Count")
axes[0].set_title("Dialogue Tokens Length")
axes[1].hist(summary_len, bins=100, color="black", edgecolor="black")
axes[1].set_xlabel("Length")
axes[1].set_ylabel("Count")
axes[1].set_title("Summary Tokens Length")
plt.show()
