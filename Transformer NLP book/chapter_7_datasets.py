import ast

from datasets import load_dataset

dataset = load_dataset("csv", data_files = \
    {"train":r"D:\.cache\datasets\SubjQA-master\SubjQA\electronics\splits\train.csv",
     "test":r"D:\.cache\datasets\SubjQA-master\SubjQA\electronics\splits\test.csv",
     "validation":r"D:\.cache\datasets\SubjQA-master\SubjQA\electronics\splits\dev.csv"})

# 处理数据，答案区间为字符串，需要转为元组
def answer_start_to_tuple(example):
    answer_start_end = ast.literal_eval(example["human_ans_indices"])
    return {"answer_start_end": answer_start_end}
dataset = dataset.map(answer_start_to_tuple)
dataset = dataset.map(lambda example: {"answer_text":  [] if example["is_ans_subjective"] == "TRUE" else example["review"][example["answer_start_end"][0]:example["answer_start_end"][1]]})
dataset = dataset.select_columns(["item_id", "question", "review", "answer_text", "answer_start_end"])
dataset = dataset.rename_column("review", "context")
print( dataset)
import os
if not os.path.exists("./datasets/subjQA/"):
    os.makedirs("./datasets/subjQA/")
for split, data in dataset.items():
    data = data.to_pandas()
    data["answer_start_end"] = data["answer_start_end"].apply(str)
    data.to_csv(f"./datasets/subjQA/datasets_{split}.csv", index=False, encoding="utf-8")





