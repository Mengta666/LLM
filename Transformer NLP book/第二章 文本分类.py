from datasets import load_dataset


dataset = load_dataset("emotion")
train_dataset = dataset["train"]
checkpoint = "distilbert-base-uncased"

# 处理数据
# 对数据进行分词
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)
dataset_encode = dataset.map(tokenize, batched=True)
print(dataset_encode["train"][0])

# 开始训练
import torch
# # 方法1：使用AutoModel
# from transformers import AutoModel
# model = AutoModel.from_pretrained(checkpoint)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # 训练模型，计算隐层
# def hidden_state(batch):
#     inputs = {k: v.to(device) for k,v in batch.items()}
#     with torch.no_grad():
#         last_hidden = model(**inputs).last_hidden_state
#     return {"hidden_state": last_hidden[:,0].cpu().numpy()}
#
# dataset_encode.set_format("torch", columns=["label", "input_ids", "attention_mask"])
# dataset_hidden = dataset_encode.map(hidden_state, batched=True)
# import numpy as np
# X_train = np.array(dataset_hidden["train"]["hidden_state"])
# X_test = np.array(dataset_hidden["test"]["hidden_state"])
# y_train = np.array(dataset_hidden["train"]["label"])
# y_test = np.array(dataset_hidden["test"]["label"])
# # 后续使用sklearn中的线性模型进行训练，不再说明
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

# # 方法2：使用AutoModelForSequenceClassification
# from transformers import AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# from transformers import Trainer, TrainingArguments
# from sklearn.metrics import accuracy_score, f1_score
# import numpy as np
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     acc = accuracy_score(predictions, labels)
#     f1 = f1_score(predictions, labels, average="weighted")
#     return {
#         "accuracy": acc,
#         "f1": f1,
#     }
# batch_size = 64
# train_step = len(train_dataset) // batch_size
# train_args = TrainingArguments(
#     output_dir="第二章 文本分类",
#     num_train_epochs=4,
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",  # 每轮保存模型
#     load_best_model_at_end=True,  # 加载验证集上最佳模型
#     metric_for_best_model="accuracy",  # 根据准确率选择最佳模型
#     warmup_steps=500,  # 学习率预热步数
#     logging_steps=10,  # 每 10 步记录一次日志
#     lr_scheduler_type="linear",  # 线性学习率调度
#     disable_tqdm=False,
# )
# trainer = Trainer(
#     model=model,
#     args=train_args,
#     train_dataset=dataset_encode["train"],
#     eval_dataset=dataset_encode["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
# trainer.train()
# predictions = trainer.predict(dataset_encode["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape, predictions)

# 方法三，使用torch自定义训练
dataset_encode.set_format("torch", columns=["input_ids", "attention_mask", "label"])
from transformers import AutoModelForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
model.to(device)
from transformers import DataCollatorWithPadding
data_collater = DataCollatorWithPadding(tokenizer=tokenizer)
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    dataset_encode["train"], shuffle=True, batch_size=8, collate_fn=data_collater
)
validation_dataloader = DataLoader(
    dataset_encode["validation"], batch_size=8, collate_fn=data_collater
)
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=5e-5)
from transformers import get_scheduler
num_epochs = 5
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
progress_bar.close()

from sklearn.metrics import f1_score, accuracy_score
model.eval()
pred_lab = []
orgin_lab = []
for batch in validation_dataloader:
    batch = {k: v.to(device) for k,v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, axis=1)
    pred_lab.extend(predictions.cpu().numpy())
    orgin_lab.extend(batch["labels"].cpu().numpy())

acc = accuracy_score(pred_lab, orgin_lab)
f1 = f1_score(pred_lab, orgin_lab, average="weighted")
print(f"Accuracy: {acc}, F1: {f1}")




