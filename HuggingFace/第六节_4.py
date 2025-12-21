import torch
from transformers import pipeline

question_answerer = pipeline("question-answering")
context = """Transformers is backed by the three most popular deep learning libraries Jax, PyTorch, and TensorFlow with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other."""
question = "Which deep learning libraries back Transformers?"

# 这个管道可以接收很长的上下文,但是假如使用 tokenizer 分词无法接收很长的上下文,需要另作处理
long_context = """
Transformers: State of the Art NLP
Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.
Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and then share them with the community on our model hub.
At the same time, each python module defining an architecture is fully standalone and can be modified to enable quick research experiments.
Why should I use transformers?
1. Easy-to-use state-of-the-art models:
- High performance on NLU and NLG tasks.
- Low barrier to entry for educators and practitioners.
- Few user-facing abstractions with just three classes to learn.
- A unified API for using all our pretrained models.
- Lower compute costs, smaller carbon footprint:
2. Researchers can share trained models instead of always retraining.
- Practitioners can reduce compute time and production costs.
- Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.
3. Choose the right framework for every part of a model's lifetime:
- Train state-of-the-art models in 3 lines of code.
- Move a single model between TF2.0/PyTorch frameworks at will.
- Seamlessly pick the right framework for training, evaluation and production.
4. Easily customize a model or an example to your needs:
- We provide examples for each architecture to reproduce the results published by its original authors.
- Model internals are exposed as consistently as possible.
- Model files can be used independently of the library for quick experiments.
Transformers is backed by the three most popular deep learning libraries Jax, PyTorch and TensorFlow with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other.
"""

# {'score': 0.9930381774902344, 'start': 1848, 'end': 1875, 'answer': 'Jax, PyTorch and TensorFlow'}
print(question_answerer(question=question, context=long_context))

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors='pt')
print(inputs.tokens())

# 使用 sequence_ids()来获取每一句中每个分词的掩码, sequence_ids 中已将将[CLS],[SEP]的掩码设置为 None
sequence_ids = inputs.sequence_ids()

# model(**inputs)返回的是 context 某个区间的概率(start和end)
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# 设置掩码,context 所有分词对应的掩码为1, 取context 的内容
print(sequence_ids)
mask = [i != 1 for i in sequence_ids]
# 不屏蔽[CLS]
mask[0] = False
# 加一个维度[None],因为 logits 是二维的
mask = torch.tensor(mask)[None]
print(mask)

# 将屏蔽的部分的分数(logits)设置为-10000,这个做softmax 后会始终保持到很低的概率
# 在PyTorch中,布尔张量可以用于索引操作 start_logits[mask] 选择 mask 中值为 True 的位置对应的 logits
start_logits[mask] = -10000
end_logits[mask] = -10000
print(start_logits, end_logits)

# 将上面的logits 转换为概率,保留为 tensor格式,得做叉乘(也就是说 start 和 end 的每个元素都要进行一次乘积)
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
print(start_probabilities, end_probabilities)

# [:, None] 在start_probabilities 的第二个维度后面增加一个维度,形状从(batch_size, sequence_length)变为(batch_size, sequence_length, 1)
# 为什么要将维度转为三维再进行运算(虽然这里的结果批次是一批,但是对于多批时,由于 start 每一批中的每一维都要和end中的每一批的每一维进行叉乘)
# 最后得到一个二维矩阵,但是加上对于批次,很显然是一个三维向量
# torch 在三维中会根据其中的值自动展开,做叉乘,最后得到一个三维向量,类似于手写二维矩阵的叉乘计算时的三层循环,如下:
# for i in range(batch_size):
#     for j in range(sequence_length_row):
#         for k in range(sequence_length_col):
#             scores[i, j, k] = start_probabilities[i, j] * end_probabilities[i, k]

scores = start_probabilities[:, None] * end_probabilities[None, :]

# 取上三角部分,下三角部分是 start 大于 end 的部分,不符合,不需要
scores = torch.triu(scores)

# 获取里面最大的值,并且返回一个索引,该索引计算方式:
# max_idx = batch * (sequence_length_row * sequence_length_col) + j * sequence_length_row + k
max_idx = scores.argmax().item()

# 求得索引所在的行和列,由于batch_size=1(即batch=0)计算简单(如下),当batch_size=2时
start_idx = max_idx // scores.shape[1]
end_idx = max_idx % scores.shape[1]
print(scores[start_idx, end_idx])

# 取出下标区间对应的句子
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_offset = offsets[start_idx]
end_offset = offsets[end_idx]

print({
    "answer": context[start_offset[0]:end_offset[1]],
    "start": start_offset[0],
    "end": end_offset[1],
    "scores": scores[start_idx, end_idx].item()
})
# 对于 long context 怎么处理?
# 使用 pipeline 可以很好地处理长文本,但是当需要微调时,仍需要自定义 tokenizer 和 model
# 分词后的长度为461,远远超过了某些模型能正常处理的长度,long context 中结果在最后面
# 将长文本分割成小块,使用tokenizer 中的参数:return_overflowing_tokens=True
# stride 参数决定新分割的块有多少个词汇是和前一块句子的最后的词汇是一样的

inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

print(tokenizer.decode(inputs["input_ids"][0]), tokenizer.decode(inputs["input_ids"][1]))

# 不需要 overflow_to_sample_mapping
_ = inputs.pop("overflow_to_sample_mapping")
# 取得每一块中每个词的偏移量
offsets = inputs.pop("offset_mapping")

# 将需要训练的输入转化为tensor
inputs = inputs.convert_to_tensors("pt")
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits

# 作处理,屏蔽第0句(这个是问题句)
sequence_ids = inputs.sequence_ids()
print(len(sequence_ids))
mask = [i != 1 for i in sequence_ids]
mask[0] = False
# 屏蔽所有的[PAD] tokens, torch.logical_or 对两个布尔张量进行逐元素逻辑或运算
# PAD 对应的是False, inputs["attention_mask"] 对应的是True, 或运算后是 True
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))
start_logits[mask] = -10000
end_logits[mask] = -10000

# 计算概率
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

# 计算区间上的概率,前面概率张量的大小:torch.Size([2, 384]),两个batch
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

# 现在已经得到了每一块的最大分数和分词对应的起始位置,结合 offsets 使用
answer = {"answer": None, "start": 0, "end": 0, "score": -1}

for candidate, offset in zip(candidates, offsets):
    if answer["score"] < candidate[2]:
        answer["score"] = candidate[2]
        start_char = offset[candidate[0]][0]
        end_char = offset[candidate[1]][1]
        answer["start"] = start_char
        answer["end"] = end_char
        answer["answer"] = long_context[start_char:end_char]

print(answer)
# {'answer': 'Jax, PyTorch and TensorFlow', 'start': 1848, 'end': 1875, 'score': 0.9930380582809448}
