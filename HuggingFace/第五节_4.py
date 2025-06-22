from datasets import load_dataset
from datasets import Dataset

# dataset = load_dataset("json", data_files="./github/huggingface/datasets-issues-with-comments-clean.jsonl", split="train")
# dataset = dataset.filter(lambda x: "pull" not in x["html_url"])
# dataset.to_json(r".\github\huggingface\datasets-issues-with-comments-only-issues.jsonl")
# from huggingface_hub import HfApi
# import os
# api = HfApi(token=os.getenv("HF_TOKEN"))
# api.upload_file(
#     path_or_fileobj=r".\github\huggingface\datasets-issues-with-comments-only-issues.jsonl",  # 本地文件路径
#     path_in_repo="datasets-issues-with-comments-only-issues.jsonl",  # 文件在仓库中的目标路径
#     repo_id="mengta666/huggingface_github",  # 仓库 ID
#     repo_type="dataset",  # 仓库类型
# )
# print(dataset)

# 使用最新清理的数据，进行文本向量嵌入
issues_dataset = load_dataset("json", data_files="./github/huggingface/datasets-issues-with-comments-only-issues.jsonl", split="train")
print(issues_dataset)
# 只保留需要的列
columns_to_keep = ["title", "body", "html_url", "comments"]
all_columns = issues_dataset.column_names
# 取一个差集
columns_to_remove = set(columns_to_keep).symmetric_difference(all_columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
print(issues_dataset)

# 对于一个问题下面可能有多个回答，需要把他展开成一个问题一个回答，使用pandas的explode函数
issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)
# 直接加载内存中的数据集
comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)
# 统计每个评论的字数，并过滤掉字数较少的评论
comments_dataset = comments_dataset.map(lambda x : {"comment_length": len(x["comments"].split())})
comments_dataset = comments_dataset.filter(lambda x : x["comment_length"] > 15)
print(comments_dataset[0])

# 将题目，说明，回答合并一下
# 过滤掉空描述的部分
comments_dataset = comments_dataset.filter(lambda x : x["body"] is not None)
comments_dataset = comments_dataset.map(lambda x : {"text" : x["title"] + "\n" +
                                                    x["body"] + "\n" + x["comments"]})
print(comments_dataset)
print(comments_dataset[0])

# 开始做文本向量嵌入：
# 有一个名为 sentence-transformers 的库专门用于创建文本嵌入。
# 这次要实现的是非对称语义搜索（asymmetric semantic search），
# 因为我们有一个简短的查询，我们希望在比如 issue 评论等更长的文档中找到其匹配的文本。
# multi-qa-mpnet-base-dot-v1 checkpoint 在语义搜索方面具有最佳性能，因此将使用它。

from transformers import AutoModel, AutoTokenizer
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

import torch
device = torch.device("cuda")
model.to(device)
# 进行文本向量嵌入
# 先使用模型对句子进行向量化
def get_embeddings(text):
    tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # print(tokenized_text)
    # 将序列化后的分词值放入gpu中
    input_text = {key : v.to(device) for key,v in tokenized_text.items()}
    output_text = model(**input_text)
    # 取最后一层的输出（也就是最后一个隐层），
    # last_hidden_state 是一个三维张量，形状为 [batch_size, sequence_length, hidden_size]
    # [:, 0] 是从 last_hidden_state 中提取 每个样本的 [CLS] token 的向量。
    # 在 Transformer 模型（如 BERT 或 MPNet）中，输入序列通常以一个特殊的 [CLS] token 开头（索引为 0），这个 token 被设计为汇总整个序列的语义信息。
    # 也就是说，其实最后一个0其实是取的第0个text（不然，text也只有一个完整的句子，序号为0，以 [CLS] token 开头，SEP 结束）
    return output_text.last_hidden_state[:,0]

# output = get_embeddings(comments_dataset[0]["text"])
# print(output.shape)
# .detach().cpu().numpy()[0] 是为了将 get_embeddings 的输出（PyTorch 张量）转换为适合 embeddings_dataset 存储的格式（NumPy 数组或 Python 列表）
# .detach() 去除tensor的梯度信息，不需要梯度，只要输出，.cpu() 将张量从GPU转到cpu，再转为numpy数组
embeddings_dataset = comments_dataset.map(lambda x : {"embeddings" : get_embeddings(x["text"]).detach().cpu().numpy()[0]})

# 使用 FAISS 进行高效的相似性搜索
# 对文本向量进行索引
embeddings_dataset.add_faiss_index(column="embeddings")
# 测试，找到相似性高的句子
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
# 以 embeddings 进行索引查找，找到最匹配的前五个，返回的是一个元组，包括评分（评价查询和文档之间的相似程度）和对应的样本（这里是 5 个最佳匹配）
scores, samples = embeddings_dataset.get_nearest_examples("embeddings",question_embedding, k=5)
print(samples)
import pandas as pd
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()

# over!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


