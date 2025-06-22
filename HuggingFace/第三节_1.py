from datasets import  load_dataset

# 导入数据集
raw_datasets = load_dataset("glue","mrpc")
print(raw_datasets["train"].features)
print(raw_datasets["train"][0])

from transformers import  AutoTokenizer
from transformers import DataCollatorWithPadding

# 预处理
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# sentence_ids = tokenizer(raw_datasets["train"][0]["sentence1"], raw_datasets["train"][0]["sentence2"], truncation=True, padding=True, batch_first=True,return_tensors="pt")
# print(sentence_ids)
# print(tokenizer.decode(sentence_ids["input_ids"][0]))
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets["train"][0])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 训练
import numpy as np
import evaluate
from transformers  import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification

training_args = TrainingArguments("test-trainer")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)
def compute_metric(eval_pred):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return metric.compute(predictions=pred, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metric
)
trainer.train()

# 评估
# 使用compute_metrics=compute_metric自动输出评估指标
# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape, predictions)


