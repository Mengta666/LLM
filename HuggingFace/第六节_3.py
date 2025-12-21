from transformers import pipeline

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
print(token_classifier(example))
# [{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
# {'entity_group': 'ORG', 'score': 0.9796019, 'word': 'Hugging Face', 'start': 33, 'end': 45},
# {'entity_group': 'LOC', 'score': 0.9932106, 'word': 'Brooklyn', 'start': 49, 'end': 57}]

# 如何利用 AutoModel 和 AutoTokenizer 来取得上述通道一样效果
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
model = AutoModelForTokenClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

input = tokenizer(example, return_tensors='pt')
outputs = model(**input)
print(outputs.logits.shape)

# 使用 softmax 函数计算概率
import torch

scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
# 使用取出最大的结果既是预测结果,返回的是索引
result = outputs.logits.argmax(dim=-1)[0].tolist()
print(scores, result)

# 将得到的结果与标签相对应,标签在 model.config.id2label 中
# 使用 return_offsets_mapping=True 参数,得到每一个分词的偏移量(也就是在实际句子中的起始位置)
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets['offset_mapping']

import numpy as np

# 合并是同一个单词的部分
idx = 0
while idx < len(result):
    pred = result[idx]
    label = model.config.id2label[pred]

    if label != 'O':
        # 不要 I-PER 中的 I-
        label = label[2:]
        start, _ = offsets[idx]
        all_score = []

        while (idx < len(result) and model.config.id2label[result[idx]] == f"I-{label}"):
            all_score.append(scores[idx][result[idx]])
            _, end = offsets[idx]
            idx += 1

        score = np.mean(all_score)
        word = example[start:end]
        results.append({
            'entity': label,
            'score': score,
            'word': word,
            'start': start,
            'end': end
        })
    idx += 1

# # 没有合并的逻辑
# for idx, pred in enumerate(result):
#     label = model.config.id2label[pred]
#     # O表示没有分类
#     if label != 'O':
#         start, end = offsets[idx]
#         results.append({
#             'entity': label,
#             'score': scores[idx][pred],
#             'word': tokens[idx],
#             'start': start,
#             'end': end
#         })
print(results)
# [{'entity': 'PER', 'score': 0.9981694370508194, 'word': 'Sylvain', 'start': 11, 'end': 18},
# {'entity': 'ORG', 'score': 0.9796018997828165, 'word': 'Hugging Face', 'start': 33, 'end': 45},
# {'entity': 'LOC', 'score': 0.9932106137275696, 'word': 'Brooklyn', 'start': 49, 'end': 57}]