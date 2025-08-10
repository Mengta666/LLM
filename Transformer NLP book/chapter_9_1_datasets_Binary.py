import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import nlpaug.augmenter.word as naw
from transformers import set_seed
from transformers import AutoModel, AutoTokenizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

baseurl = "https://git.io/nlp-with-transformers"
df_issues = pd.read_json(baseurl, lines=True, convert_dates=False)
# 手动转换时间戳列，指定单位为毫秒
date_columns = ['created_at', 'updated_at']
for col in date_columns:
    df_issues[col] = pd.to_datetime(df_issues[col], unit='ms', errors='coerce')
# (9930, 26)
print(df_issues.shape)
# Index(['url', 'repository_url', 'labels_url', 'comments_url', 'events_url',
#        'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels',
#        'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments',
#        'created_at', 'updated_at', 'closed_at', 'author_association',
#        'active_lock_reason', 'body', 'performed_via_github_app',
#        'pull_request'],
#       dtype='object')
print(df_issues.columns)
cols = ["url", "id", "title", "user", "labels", "state", "created_at", "body"]
print(df_issues.loc[2, cols].to_frame())
# 将标签转换成数组列出
df_issues["labels"] = df_issues["labels"].apply(lambda x : [label["name"] for label in x])
# df.loc[row_labels, column_labels]
# row_labels：行索引（可以是单一索引、索引列表、切片或布尔掩码）。
# column_labels：列名（可以是单一列名、列名列表、切片）。
print(df_issues.loc[2, cols].to_frame())
# labels     0     1    2    3   4  5
# count   6440  3057  305  100  25  3
print(df_issues["labels"].apply(lambda x : len(x)).value_counts().to_frame().T)
# 统计每一种标签对应的数据数量
df_counts = df_issues["labels"].explode().value_counts()
# labels  wontfix  model card  Core: Tokenization  ...  work in progress  wandb  fp16
# count      2284         649                 106  ...                 1      1     1
# [1 rows x 65 columns]
# 数据标注下的种类只有65种，并且大多数数据甚至没有标注
print(df_counts.to_frame().T)

# 将需要的标签名称进行标准化处理
labels_map = {
    "Core: Tokenization": "tokenization",
    "New model": "new model",
    "Core: Modeling": "model training",
    "Usage": "usage",
    "Core: Pipeline": "pipeline",
    "TensorFlow": "tensorflow or tf",
    "PyTorch": "pytorch",
    "Examples": "examples",
    "Documentation": "documentation"
}
df_issues["labels"] = df_issues["labels"].apply(
    lambda x : [ labels_map[label] for label in x if label in labels_map]
)
df_counts = df_issues["labels"].explode().value_counts()
# labels  tokenization  new model  ...  documentation  examples
# count            106         98  ...             28        24
# [1 rows x 9 columns]
print(df_counts.to_frame().T)

# 将其他的数据集标记为未标注
df_issues["split"] = "unlabeled"
mask = df_issues["labels"].apply(lambda x : len(x) > 0)
# 这里 loc[mask, "split"] 根据布尔掩码 mask（表示 labels 列非空的行）将 split 列的值设置为 "labeled"。
df_issues.loc[mask, "split"] = "labeled"
# unlabeled   9489
# labeled      441
print(df_issues["split"].value_counts().to_frame())

# 将title和body合并为text
df_issues["text"] = df_issues.apply(lambda x: x["title"] + "\n\n" + x["body"], axis=1)

# 删除重复文本的数据
df_issues = df_issues.drop_duplicates(subset="text")


# 创建数据集
# 创建独热编码
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
all_labels = list(labels_map.values())
mlb.fit([all_labels])

from skmultilearn.model_selection import iterative_train_test_split
def balanced_split(df, test_size=0.5):
    # np.expand_dims(..., axis=1)：将一维数组（如 [0, 1, 2, ...]）转换为二维数组（如 [[0], [1], [2], ...]），形状为 (n_samples, 1)。
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    # mlb.transform(df["labels"])：将 df["labels"]（多标签列表，如 [['tokenization'], ['new model', 'usage']]) 转换为二值矩阵，形状为 (n_samples, n_labels)。
    feature = mlb.transform(df["labels"])
    # ind_train 和 ind_test 是 iterative_train_test_split 返回的索引数组
    ind_train,_,ind_test,_ = iterative_train_test_split(ind, feature, test_size=test_size)
    # df.iloc[row_indices, column_indices]
    # row_indices: 行的整数索引（可以是单一整数、整数列表、切片或布尔数组）。
    # column_indices: 列的整数位置（可以是单一整数、整数列表、切片）。
    return df.iloc[ind_train[:,0]], df.iloc[ind_test[:,0]]

from sklearn.model_selection import train_test_split
np.random.seed(0)
df_clean = df_issues[["text", "labels", "split"]].reset_index(drop=True).copy()
df_unsup = df_clean.loc[df_clean["split"] == "unlabeled", ["text", "labels"]]
df_sup = df_clean.loc[df_clean["split"] == "labeled", ["text", "labels"]]
df_train, df_tmp = balanced_split(df_sup)
df_val, df_test = balanced_split(df_tmp)

# 转换成dataset的数据格式，方便在transformers中使用
from datasets import Dataset, DatasetDict
ds = DatasetDict({
    "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
    "val": Dataset.from_pandas(df_val.reset_index(drop=True)),
    "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    "unsup": Dataset.from_pandas(df_unsup.reset_index(drop=True))
})

print(ds)
print(len(ds["train"]))

# 由于训练的数据集只有220个有标注的数据集，首先对训练集进行分片，训练一个基线模型
np.random.seed(0)
all_indices = np.expand_dims(np.arange(len(ds["train"])), axis=1)
indices_pool = all_indices
labels = mlb.transform(ds["train"]["labels"])
# 按照比例进行分片
train_example = [8, 16, 32, 64, 128]
train_slices, last_k= [], 0

for i, k in enumerate(train_example):
    indices_pool, labels, new_slice, _ = iterative_train_test_split(indices_pool, labels,  test_size=(k-last_k)/len(labels))
    last_k = k
    if i==0:
        train_slices.append(new_slice)
    else:
        train_slices.append(np.concatenate((train_slices[-1], new_slice)))

train_example.append(len(ds["train"]))
train_slices.append(all_indices)
# np.squeeze(train_slice) 降维，删除维度为1的维
train_slices = [ np.squeeze(train_slice) for train_slice in train_slices]
# [8, 16, 32, 64, 128, 223]
# [10, 19, 36, 68, 134, 223]
print(train_example)
print([len(train_slice) for train_slice in train_slices])

# 将数据集中的标签进行one-hot编码
def one_hot(example):
    return {
        "label_ids": mlb.transform(example["labels"])
    }
ds = ds.map(one_hot, batched=True)
print(ds)

# 构建一个基线模型-朴素贝叶斯分类器
macro_scores, micro_scores = defaultdict(list), defaultdict(list)
for train_slice in train_slices:
    ds_train_sample = ds["train"].select(train_slice)
    y_train = np.array(ds_train_sample["label_ids"])
    y_test = np.array(ds["test"]["label_ids"])
    # 词袋模型
    count_vect = CountVectorizer()
    x_train = count_vect.fit_transform(ds_train_sample["text"])
    x_test = count_vect.transform(ds["test"]["text"])
    # 使用朴素贝叶斯分类预测
    classificer = BinaryRelevance(classifier=MultinomialNB())
    # 激活模型
    classificer.fit(x_train, y_train)
    y_test_pre = classificer.predict(x_test)
    clf_report = classification_report(y_test, y_test_pre, output_dict=True, target_names=mlb.classes_, zero_division=0)
    macro_scores["Naive Bayes"].append(clf_report["macro avg"]["f1-score"])
    micro_scores["Naive Bayes"].append(clf_report["micro avg"]["f1-score"])

print(macro_scores)
print(micro_scores)

# 宏观 F1 分数：如果稀有标签（如 examples 或 documentation）的预测性能较差，宏观 F1 分数会较低，反映模型在小类标签上的不足。
# 宏观 F1 分数，显示各标签的平均性能。
# 微观 F1 分数：由于微观 F1 分数更关注整体预测的正确性，通常在标签分布不均衡时（如本例中某些标签如 tokenization 出现 106 次，而 examples 仅 24 次），它会更倾向于反映常见标签的性能。
# 微观 F1 分数，显示整体预测性能。
# 绘制宏观和微观的F1-score
def plot_metrics(macro_scores, micro_scores, sample_size, current_model):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for run in macro_scores.keys():
        if run == current_model:
            ax0.plot(sample_size, macro_scores[run], label=run, linewidth=2, linestyle="--")
            ax1.plot(sample_size, micro_scores[run], label=run, linewidth=2, linestyle="-.")
        else:
            ax0.plot(sample_size, macro_scores[run], label=run)
            ax1.plot(sample_size, micro_scores[run], label=run)

    ax0.set_title("Macro F1-score")
    ax1.set_title("Micro F1-score")
    for ax in [ax0, ax1]:
        ax.set_xlabel("Training sample size")
        ax.set_ylabel("F1-score")
        ax.set_xticks(sample_size)
        ax.legend()
    plt.legend(loc="best")
    plt.show()
plot_metrics(macro_scores, micro_scores, train_example, "Naive Bayes")


# 零样本学习
# 使用MNLI模型进行零样本分类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline("zero-shot-classification", device=device)
output = pipe(
    ds["train"][0]["text"], all_labels, multi_label=True
)
# new model: 0.984
# tensorflow or tf: 0.366
# examples: 0.336
# usage: 0.297
# pytorch: 0.248
# documentation: 0.245
# model training: 0.235
# tokenization: 0.174
# pipeline: 0.160
for label, score in zip(output["labels"], output["scores"]):
    print(f"{label}: {score:.3f}")
# Add new CANINE model
#
# # 🌟 New model addition
#
# ## Model description
#
# Google recently proposed a n
print(output["sequence"][:100])

# 往验证集里面加入预测后的标签列表，按照分数进行排序
def zero_shot_pipeline(example):
    output = pipe(
        example["text"], all_labels, multi_label=True
    )
    return {
        "predicted_labels": output["labels"],
        "scores": output["scores"]
    }
ds_zero_shot = ds["val"].map(zero_shot_pipeline)
print(ds_zero_shot)

# 获取预测的标签，两种方式：1. 一个下限，超过的标签全部选择，2. top_K
def get_preds(example, top_k=None, threshold=None):
    preds = []
    if threshold:
        for label, score in zip(example["predicted_labels"], example["scores"]):
            if score > threshold:
                preds.append(label)
    elif top_k:
        preds = example["predicted_labels"][:top_k]
    else:
        raise ValueError("Please specify either top_k or threshold")
    return {
        # 一维列表
        "pred_label_ids": list(np.squeeze(mlb.transform([preds])))
    }

# 计算f1分数
def get_clf_report(ds):
    y_true = np.array(ds["label_ids"])
    y_pred = np.array(ds["pred_label_ids"])
    return classification_report(
        y_true, y_pred, output_dict=True, target_names=mlb.classes_, zero_division=0
    )
macros, micros = [], []
for top_k in [1, 2, 3, 4]:
    ds_zero_shot = ds_zero_shot.map(get_preds, batched=False, fn_kwargs={"top_k": top_k})
    clf_report = get_clf_report(ds_zero_shot)
    macros.append(clf_report["macro avg"]["f1-score"])
    micros.append(clf_report["micro avg"]["f1-score"])
# 绘图
plt.plot([1, 2, 3, 4], micros, label="Micro F1-score", linestyle="--")
plt.plot([1, 2, 3, 4], macros, label="Macro F1-score", linestyle="-.")
plt.legend()
plt.show()

macros, micros= [], []
for threshold in np.linspace(0.1, 1.0, 100):
    ds_zero_shot = ds_zero_shot.map(get_preds, batched=False, fn_kwargs={"threshold": threshold})
    clf_report = get_clf_report(ds_zero_shot)
    macros.append(clf_report["macro avg"]["f1-score"])
    micros.append(clf_report["micro avg"]["f1-score"])
plt.plot(np.linspace(0.1, 1.0, 100), micros, label="Micro F1-score", linestyle="--")
plt.plot(np.linspace(0.1, 1.0, 100), macros, label="Macro F1-score", linestyle="-.")
plt.legend()
plt.show()
the_best_threshold_micro = np.linspace(0.1, 1.0, 100)[np.argmax(micros)]
# The best threshold micro is 0.76
print(f"The best threshold micro is {the_best_threshold_micro:.2f}")
the_best_threshold_macro = np.linspace(0.1, 1.0, 100)[np.argmax(macros)]
# The best threshold macro is 0.73
print(f"The best threshold macro is {the_best_threshold_macro:.2f}")

# 选择top_k = 1 与朴素贝叶斯进行比较
ds_zero_shot = ds_zero_shot.map(get_preds, batched=False, fn_kwargs={"top_k": 1})
clf_report = get_clf_report(ds_zero_shot)
for i in range(len(train_example)):
    macro_scores["top_k"].append(clf_report["macro avg"]["f1-score"])
    micro_scores["top_k"].append(clf_report["micro avg"]["f1-score"])
plot_metrics(macro_scores, micro_scores, train_example, "Naive Bayes")



# 少样本学习
set_seed(3)
aug = naw.ContextualWordEmbsAug(model_path="distilbert-base-uncased", action="substitute", device="cuda")
text = "Transformers are the most popular toys"
# Transformers are the most popular toys
print(text)
# transformers have the most available toys
print(aug.augment(text)[0])
# are 被同义替换成了have
# 下面将全部数据集做一个同义替换
def augmnet_text(batch, transformation_per_example=1):
    text_aug, label_ids = [], []
    for text, label in zip(batch["text"], batch["label_ids"]):
        text_aug+=[text]
        label_ids+=[label]
        for _ in range(transformation_per_example):
            text_aug+=aug.augment(text)
            label_ids+=[label]
    return {
        "text": text_aug,
        "label_ids": label_ids
    }
ds_train_sample = ds_train_sample.map(augmnet_text, batched=True, remove_columns=ds_train_sample.column_names).shuffle(42)
# (446, 2)
print(ds_train_sample.shape)
# 用新的数据集训练朴素贝叶斯模型后再评估
for train_slice in train_slices:
    ds_train_sample_aug = ds_train_sample.select(train_slice)
    y_train = np.array(ds_train_sample_aug["label_ids"])
    y_test = np.array(ds["test"]["label_ids"])
    count_vect = CountVectorizer()
    x_train = count_vect.fit_transform(ds_train_sample_aug["text"])
    x_test = count_vect.transform(ds["test"]["text"])
    # 使用朴素贝叶斯分类预测
    classificer = BinaryRelevance(classifier=MultinomialNB())
    # 激活模型
    classificer.fit(x_train, y_train)
    y_test_pre = classificer.predict(x_test)
    clf = classification_report(y_test, y_test_pre, output_dict=True, target_names=mlb.classes_, zero_division=0)
    macro_scores["Naive Bayes Aug"].append(clf["macro avg"]["f1-score"])
    micro_scores["Naive Bayes Aug"].append(clf["micro avg"]["f1-score"])
plot_metrics(macro_scores, micro_scores, train_example, "Naive Bayes Aug")


# 平均汇聚
checkpoint = "miguelvictor/python-gpt2-large"
model = AutoModel.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device=device)
# 设置填充词元
tokenizer.pad_token = tokenizer.eos_token

def mean_pooling(model_output, attention_mask):
    output_embeddings = model_output[0]
    input_mask = attention_mask.unsqueeze(-1).expand(output_embeddings.size()).float()
    # torch.Size([1, 64, 1280])
    # print(input_mask.shape)
    # 逐元素相乘
    embeddings_sum = torch.sum(output_embeddings * input_mask, 1)
    mask_sum = torch.clamp(input_mask.sum(1), min=1e-9)
    # torch.Size([1, 1280])
    # print(embeddings_sum.shape)
    return embeddings_sum / mask_sum
def embed_text(batch):
    input_ids = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
    pool_embeddings = mean_pooling(model(**input_ids), input_ids["attention_mask"])
    return {
        "embedding": pool_embeddings.detach().cpu().numpy()
    }
embs_train = ds["train"].map(embed_text, batched=True, batch_size=8)
embs_test = ds["test"].map(embed_text, batched=True, batch_size=8)
embs_valid = ds["val"].map(embed_text, batched=True, batch_size=8)

# 创建一个叫做embedding的索引
embs_train.add_faiss_index("embedding")
# 设置验证集序号i,和检索的最近邻数量k
i, k = 0, 3
query = np.array(embs_valid[0]["embedding"], dtype=np.float32)
# 调用get_nearest_examples方法来做最近邻搜索
scores, samples = embs_train.get_nearest_examples("embedding", query, k=k)
print(f"query text:\n{embs_valid[i]['text'][:100]}")
print(f"query labels: {embs_valid[i]['labels']}")
print("="*50)
for score, text, label in zip(scores, samples["text"], samples["labels"]):
    print(f"score: {score:.4f}")
    print(f"text:\n{text[:100]}")
    print(f"labels: {label}")
    print("="*50)
# # 输出：
# query text:
# Implementing efficient self attention in T5
# # 🌟 New model addition
# My teammates and I (including
# query labels: ['new model']
# ==================================================
# score: 32.0094
# text:
# Add FAVOR+ / Performer attention
# # 🌟 FAVOR+ / Performer attention addition
# Are there any plans t
# labels: ['new model']
# ==================================================
# score: 34.9671
# text:
# Implement DeLighT: Very Deep and Light-weight Transformers
# # 🌟 New model addition
# ## Model descr
# labels: ['new model']
# ==================================================
# score: 41.6796
# text:
# Funnel Transformers
# # 🌟 New model addition
# Funnel-Transformer
# ## Model description
# Funnel-Tran
# labels: ['new model']
# ==================================================


# 寻找最优的超参数k(最优的近邻个数)和m(近邻中同类标签的数量)
def get_sample_preds(sample, m):
    return (np.sum(np.array(sample["label_ids"]), axis=0) >= m).astype(int)
def find_best_k_m(ds_train, valid_queries, valid_labels, max_k=17):
    max_k = min(max_k, len(ds_train))
    perf_micro = np.zeros([max_k, max_k])
    perf_macro = np.zeros([max_k, max_k])
    for k in range(1, max_k):
        for m in range(1, k+1):
            _, samples = ds_train.get_nearest_examples_batch("embedding", valid_queries, k=k)
            preds = [ get_sample_preds(sample, m) for sample in samples]
            clf_report = classification_report(valid_labels, preds, output_dict=True, zero_division=0, target_names=mlb.classes_)
            perf_macro[k, m] = clf_report["macro avg"]["f1-score"]
            perf_micro[k, m] = clf_report["micro avg"]["f1-score"]
    return perf_micro, perf_macro
valid_queries = np.array(embs_valid["embedding"], dtype=np.float32)
valid_labels = np.array(embs_valid["labels"])
perf_micro, perf_macro = find_best_k_m(embs_train, valid_queries, valid_labels)
fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,3.5))
ax0.imshow(perf_micro)
ax1.imshow(perf_macro)
ax0.set_title("Micro-averaged F1-score")
ax1.set_title("Macro-averaged F1-score")
ax0.set_ylabel("k")
for ax in (ax0, ax1):
    ax.set_xlim(0.5,17-0.5)
    ax.set_ylim(17-0.5,0.5)
    ax.set_xlabel("m")
plt.show()

# argmax() 返回 perf_micro 中最大值的一维索引（扁平化后的索引）
# indices：整数或整数数组，表示一维数组中的索引。(这里是最大值的索引）
# shape：元组，指定多维数组的形状（例如 (rows, cols) 或 (depth, rows, cols)）。
# indices 必须是有效的索引，不能超过 shape 对应数组的元素总数（np.prod(shape)）。
# 如果 indices 是数组，返回的坐标数组形状与 indices 一致。
# 常用于处理 argmax 或 argmin 的结果，找到最大/最小值在多维数组中的位置。
k, m = np.unravel_index(perf_micro.argmax(), perf_micro.shape)
print(m,k)

# 取消embs_train中名为embedding的faiss索引，在每一个train_slice中添加它
embs_train.drop_index("embedding")
test_labels = np.array(embs_test["label_ids"])
test_queries = np.array(embs_test["embedding"], dtype=np.float32)

for train_slice in train_slices:
    embs_train_tmp = embs_train.select(train_slice)
    embs_train_tmp.add_faiss_index("embedding")
    perf_micro, _ = find_best_k_m(embs_train_tmp, test_queries, test_labels)
    k, m = np.unravel_index(perf_micro.argmax(), perf_micro.shape)
    scores, samples = embs_train_tmp.get_nearest_examples_batch("embedding", test_queries, k=int(k))
    y_pred =np.array([ get_sample_preds(sample, m) for sample in samples])
    clf_report = classification_report(test_labels, y_pred, output_dict=True, zero_division=0, target_names=mlb.classes_)
    macro_scores["Embedding"].append(clf_report["macro avg"]["f1-score"])
    micro_scores["Embedding"].append(clf_report["micro avg"]["f1-score"])
plot_metrics(macro_scores, micro_scores, train_example, "Embedding")


# 前面都是使用的一些1特殊方法对预训练的模型直接使用，现在利用已有的标注训练集进行训练
from transformers import AutoModelForSequenceClassification, AutoConfig
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)
ds_enc = ds.map(tokenize, batched=True)
ds_enc = ds_enc.remove_columns(["text", "labels"])
# 将label_ids转换成torch的浮点型（原本是np的float）
ds_enc.set_format("torch")
ds_enc = ds_enc.map(lambda x: {"label_ids_f": x["label_ids"].to(torch.float)}, remove_columns=["label_ids"])
ds_enc = ds_enc.rename_column("label_ids_f", "label_ids")
# 设置Trainer，args
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
args = TrainingArguments(
    output_dir="./chapter_9_output",
    num_train_epochs=20,
    learning_rate=3e-5,
    lr_scheduler_type="constant",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro-f1",
    save_total_limit=2,
)
from scipy.special import expit as sigmoid
def compute_metrics(pred):
    y_true = pred.label_ids
    logits = pred.predictions
    probs = sigmoid(logits)
    y_pred = (probs > 0.5).astype(float)
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, target_names=all_labels)
    print(clf_report)
    return {
        "micro-f1": clf_report["micro avg"]["f1-score"],
        "macro-f1": clf_report["macro avg"]["f1-score"],
    }
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
config = AutoConfig.from_pretrained(checkpoint, num_labels=len(all_labels))
config.problem_type = "multi_label_classification"
for train_slice in train_slices:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_enc["train"].select(train_slice),
        eval_dataset=ds_enc["val"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()
    pred = trainer.predict(ds_enc["test"])
    metrics = compute_metrics(pred)
    macro_scores["Fine-tune(vanilla)"].append(metrics["macro-f1"])
    micro_scores["Fine-tune(vanilla)"].append(metrics["micro-f1"])

plot_metrics(macro_scores, micro_scores, train_example, "Fine-tune(vanilla)")


# from transformers import DataCollatorForLanguageModeling, set_seed
# from transformers import AutoTokenizer
#
#
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# set_seed(3)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# data_collator.return_tensors = "np"
# inputs = tokenizer("Transformers are awesome!", return_tensors="pt")
# outputs = data_collator([{"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}])
# print(outputs)
# import pandas as pd
# #                      0             1     2        3       4       5
# # Original tokens  [CLS]  transformers   are  awesome       !   [SEP]
# # Masked tokens    [CLS]  transformers   are  awesome  [MASK]  [MASK]
# # Original ids       101         19081  2024    12476     999     102
# # Masked ids         101         19081  2024    12476     103     103
# # labels            -100          -100  -100     -100     999     102
# print(
#     pd.DataFrame({
#         "Original tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
#         "Masked tokens": tokenizer.convert_ids_to_tokens(outputs["input_ids"][0][0]),
#         "Original ids": inputs["input_ids"][0],
#         "Masked ids": outputs["input_ids"][0][0],
#         "labels": outputs["labels"][0][0],
#     }).T
# )

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length", return_special_tokens_mask=True)
mlm_ds = ds.map(tokenize, batched=True)
ds_mlm = mlm_ds.remove_columns(["text", "labels", "label_ids"])
from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM
data_collator_model = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
data_collator_model.return_tensors = "pt"
args_train = TrainingArguments(
    output_dir="./chapter_9_output_mask",
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    num_train_epochs=16,
    log_level="error",
    report_to="none",
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    weight_decay=0.01,
    warmup_steps=100,
)
trainer = Trainer(
    model=AutoModelForMaskedLM.from_pretrained(checkpoint),
    args=args_train,
    data_collator=data_collator_model,
    tokenizer=tokenizer,
    train_dataset=ds_mlm["unsup"],
    eval_dataset=ds_mlm["train"]
)
trainer.train()
trainer.save_model()
df_log = pd.DataFrame(trainer.state.log_history)
df_log.dropna(subset=["eval_loss"]).reset_index()["eval_loss"].plot(label="Validation")
df_log.dropna(subset=["loss"]).reset_index()["loss"].plot(label="Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

model_check = "./chapter_9_output_mask"
config = AutoConfig.from_pretrained(model_check, num_labels=len(all_labels), problem_type="multi_label_classification")
for train_slice in train_slices:
    model = AutoModelForSequenceClassification.from_pretrained(model_check, config=config)
    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=ds_enc["train"].select(train_slice),
        eval_dataset=ds_enc["val"],
    )
    trainer.train()
    pred = trainer.predict(ds_enc["test"])
    metrics = compute_metrics(pred)
    macro_scores["Fine-tune(DA)"].append(metrics["macro-f1"])
    micro_scores["Fine-tune(DA)"].append(metrics["micro-f1"])

plot_metrics(macro_scores, micro_scores, train_example, "Fine-tune(DA)")





