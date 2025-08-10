import pandas as pd
from datasets import load_dataset
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
sample_data = dataset["test"].shuffle(seed=42).select(range(300))
print(sample_data.features)

import evaluate
rouge_compute = evaluate.load("rouge")

def get_batch_text(batch, sample_text):
    for i in range(0, len(sample_text), batch):
        yield sample_text[i : i+batch]

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_summaris_pegasus(dataset, model, tokenizer, batch_size=8, device=device):
    article_batches = list(get_batch_text(batch_size, dataset["article"]))
    highlights_batches = list(get_batch_text(batch_size, dataset["highlights"]))
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(len(dataset["article"])), desc="Generating summaries")
    highlights = []
    summaries = []
    for article_batch, highlight_batch in zip(article_batches, highlights_batches):
        input_ids = tokenizer(
            article_batch,
            max_length=512,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ).to(device)
        # length_penalty=0.8 表示模型倾向于生成较短的序列
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                length_penalty=0.8,
                num_beams=4,
                max_length=128,
                use_cache=False  # 禁用缓存
            )
        highlights.extend(highlight_batch)
        summaries.extend([(tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True)).replace("<n>", " ") for i in output])
        progress_bar.update(len(article_batch))
    progress_bar.close()
    print("开始计算rouge值:")
    result = rouge_compute.compute(
        predictions=summaries,
        references=highlights
    )
    return result

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
check_point = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(check_point)
model = AutoModelForSeq2SeqLM.from_pretrained(check_point).to(device)
result = evaluate_summaris_pegasus(sample_data, model, tokenizer, batch_size=8)
print(pd.DataFrame.from_dict(result, orient="index", columns=["value"]))
# 300个样本，跑了9h
#               value
# rouge1     0.420028
# rouge2     0.199576
# rougeL     0.298535
# rougeLsum  0.359704