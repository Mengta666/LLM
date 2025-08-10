from datasets import load_dataset
from haystack import Document, Pipeline
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.utils.device import ComponentDevice
from transformers import AutoTokenizer
from elasticsearch import Elasticsearch

# 加载数据集，指定训练、测试和验证集的 CSV 文件路径
dataset = load_dataset("csv", data_files={
    "train": r".\datasets\subjQA\datasets_train.csv",
    "test": r".\datasets\subjQA\datasets_test.csv",
    "validation": r".\datasets\subjQA\datasets_validation.csv"
})
# 打印每个数据集分区的大小
print("Dataset sizes:", {split: len(dataset[split]) for split in dataset})

# 初始化 Elasticsearch 客户端，连接本地 Elasticsearch 实例
es_client = Elasticsearch(
    hosts="https://127.0.0.1:9200/",
    basic_auth=("elastic", "aefYxWUqHFJiuu-l0lvK"),  # 用户名和密码
    verify_certs=False  # 不验证 SSL 证书
)

# 定义新索引名称
new_index_name = "new_default_index"

# 定义 Elasticsearch 索引的映射，设置嵌入向量维度为 768（匹配 bert-base-uncased）
new_mapping = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},  # 文本内容字段
            "embedding": {
                "type": "dense_vector",  # 嵌入向量字段
                "dims": 768,  # 向量维度
                "index": True,  # 启用索引
                "similarity": "cosine"  # 使用余弦相似度
            },
            "meta": {  # 元数据字段
                "properties": {
                    "item_id": {"type": "keyword"},  # 产品 ID
                    "question": {"type": "text"},  # 问题
                    "answer": {"type": "text"}  # 答案
                }
            }
        }
    }
}

# 创建新索引：如果索引已存在，先删除再重新创建
if es_client.indices.exists(index=new_index_name):
    es_client.indices.delete(index=new_index_name)
es_client.indices.create(index=new_index_name, body=new_mapping)
print(f"New index '{new_index_name}' created with 768 dimensions.")

# 初始化 Elasticsearch 文档存储，连接到指定索引
document_store = ElasticsearchDocumentStore(
    hosts="https://127.0.0.1:9200/",
    basic_auth=("elastic", "aefYxWUqHFJiuu-l0lvK"),
    verify_certs=False,
    index=new_index_name,
)

# 清空文档存储中的现有文档
all_documents = document_store.filter_documents()
all_document_ids = [doc.id for doc in all_documents]
document_store.delete_documents(document_ids=all_document_ids)
# Total documents in store: 0
print("Total documents in store:", document_store.count_documents())

# 初始化文档嵌入器，使用 BERT 模型并指定运行在 GPU 上
check_point = "google-bert/bert-base-uncased"
device = ComponentDevice.from_str("cuda")
document_embedding = SentenceTransformersDocumentEmbedder(model=check_point, device=device)
document_embedding.warm_up()  # 预热嵌入器

# 定义数据集批量迭代器，按指定 batch_size 分批处理
def get_datasets_iter(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        yield batch

# 初始化 BERT 分词器
tokenizer = AutoTokenizer.from_pretrained(check_point)

documents = []
batch_size = 100
# 处理数据集，将数据转换为 Haystack Document 对象并嵌入
for split in dataset:
    print(f"Processing split: {split}, size: {len(dataset[split])}")
    for batch in get_datasets_iter(dataset[split], batch_size=batch_size):
        for i in range(len(batch["item_id"])):
            if not batch["context"][i]:  # 检查是否为空上下文
                print(f"Empty context in {split}, index {i}")
                continue
            # 对上下文进行分词，设置最大长度为 512，支持截断和滑动窗口
            inputs = tokenizer(batch["context"][i], max_length=512, truncation=True, padding="max_length", stride=64, return_overflowing_tokens=True)
            for input_ids in inputs["input_ids"]:
                doc_text = tokenizer.decode(input_ids, skip_special_tokens=True)  # 解码为文本
                documents.append(Document(
                    content=doc_text,
                    meta={"item_id": batch["item_id"][i], "question": batch["question"][i], "answer": batch["answer_text"][i]}
                ))

# 生成文档嵌入并写入文档存储，跳过重复文档
document_with_embedding = document_embedding.run(documents)
document_store.write_documents(document_with_embedding.get("documents"), policy=DuplicatePolicy.SKIP)

# 检查文档存储中的文档数量
# Total documents in store: 3312
print(f"Total documents in store: {document_store.count_documents()}")

# 创建检索管道
pipelines = Pipeline()
# 添加文本嵌入器组件，使用 BERT 模型
pipelines.add_component("text_embedder", SentenceTransformersTextEmbedder(model=check_point))
# 添加 Elasticsearch 检索器组件
pipelines.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store))
# 连接文本嵌入器和检索器
pipelines.connect("text_embedder.embedding", "retriever.query_embedding")

# 获取测试集中的第一个问题并打印相关信息
query = dataset["test"][0]["question"]
print(dataset["test"][0]["item_id"], dataset["test"][0]["context"])
#

# 运行检索管道，获取与查询相关的文档
result = pipelines.run({
    "text_embedder": {"text": query},
})
print(result["retriever"]["documents"][0])  # 打印检索到的第一个文档
# Document(id=87d5ff127d8186532f03fa83560efab93742600803c05b9eea2fb6597ff678b0, content: 'this thing gets nice and loud! the only issue i have is the proprietary connection to charge the uni...', meta: {'item_id': 'B004HHICKC', 'question': 'How is it the people there?', 'answer': 'ANSWERNOTFOUND'}, score: 0.7822685, embedding: vector of size 768)

# 初始化阅读器组件，使用 MiniLM 模型进行答案提取
from haystack.components.readers import ExtractiveReader
reader = ExtractiveReader(model="deepset/minilm-uncased-squad2", device=device)
reader.warm_up()  # 预热阅读器

# 运行阅读器，提取答案
# What is strap?
# {'answers': [ExtractedAnswer(query='What is strap?', score=0.6831972599029541, data='solid usb cable', document=Document(id=db83d7403b21d0543ba26b621f7281a2f6efd9c25e8c9e242817c391ca7c5b08, content: 'what more can you say about micra digital usb a to usb b cable other than it has a sturdy and thick ...', meta: {'item_id': 'B004CLYEFK', 'question': 'What is the length of the cable of a television ?', 'answer': 'ANSWERNOTFOUND'}, score: 0.7783623, embedding: vector of size 768), context=None, document_offset=ExtractedAnswer.Span(start=181, end=196), context_offset=None, meta={}), ExtractedAnswer(query='What is strap?', score=0.6601563692092896, data='small', document=Document(id=eeb752f50c7c816e8dae8d49e533aec23261258a3e0efc14edce97fd6aa1c0ed, content: 'this product is amazing altho i never seem to remember where i put it, because it is so small... rec...', meta: {'item_id': 'B005FYNSPK', 'question': 'What is the cost of this cellphone?', 'answer': 'ANSWERNOTFOUND'}, score: 0.78157806, embedding: vector of size 768), context=None, document_offset=ExtractedAnswer.Span(start=88, end=93), context_offset=None, meta={}), ExtractedAnswer(query='What is strap?', score=0.10766339343902587, data=None, document=None, context=None, document_offset=None, context_offset=None, meta={})]}
print(query)
read_result = reader.run(query=query, documents=result["retriever"]["documents"], top_k=2)
print(read_result)  # 打印提取的答案

# 有多种评估方法：https://docs.haystack.deepset.ai/docs/evaluators
# AnswerExactMatchEvaluator	使用真实标签评估 Haystack 流程预测的答案。它会逐个字符地检查预测答案是否与真实答案完全匹配。
# ContextRelevanceEvaluator	使用 LLM 来评估是否可以从提供的上下文推断出生成的答案。
# DeepEvalEvaluator	使用 DeepEval 评估生成管道。
# DocumentMAPEvaluator	使用 Haystack 管道检索到的文档。它会检查检索到的文档列表在多大程度上仅包含真实标签中指定的相关文档，或包含非相关文档。
# DocumentMRREvaluator	使用 Haystack 管道检索到的文档。它会检查真实标签文档在检索到的文档列表中的排名。
# DocumentNDCGEvaluator	使用 Haystack 管道检索到的文档。它会检查标准真值文档在检索到的文档列表中的排名。该指标称为归一化折扣累积增益 (NDCG)。
# DocumentRecallEvaluator	使用 Haystack 管道检索到的文档。它会检查检索到的文档数量。
# FaithfulnessEvaluator	使用LLM评估生成的答案是否可以从提供的上下文推断出来。不需要基本事实标签。
# LLMEvaluator	使用 LLM 根据包含用户定义的指令和示例的提示来评估输入。
# RagasEvaluator	使用 Ragas 框架来评估检索增强生成管道。
# SASEvaluator	使用 Haystack 管道预测的答案。它使用经过微调的语言模型检查预测答案与基本事实答案的语义相似度。
# 查阅官方文档即可
# 评估检索器，使用 DocumentMAPEvaluator
from haystack.components.evaluators import DocumentMAPEvaluator
pipelines.add_component("evaluator", DocumentMAPEvaluator())
# 针对单个查询进行评估（与原代码一致）
query = dataset["test"][0]["question"]
ground_truth_document = Document(
    content=dataset["test"][0]["context"],
    meta={
        "item_id": dataset["test"][0]["item_id"],
        "question": dataset["test"][0]["question"],
        "answer": dataset["test"][0]["answer_text"]
    }
)
result = pipelines.run({
    "text_embedder": {"text": query},
    "evaluator": {
        "ground_truth_documents": [[ground_truth_document]],
        "retrieved_documents": [result["retriever"]["documents"][:5]]
    }
})

# 0.0
print(result["evaluator"]["score"])

# 评估阅读器，使用AnswerExactMatchEvaluator （EM 精准匹配）
from haystack.components.evaluators import AnswerExactMatchEvaluator
answer_match_evaluator = AnswerExactMatchEvaluator()
result = answer_match_evaluator.run(
    predicted_answers=[read_result["answers"]],
    ground_truth_answers=[dataset["test"][0]["answer_text"]]
)
# 0.0
print(result["score"])






