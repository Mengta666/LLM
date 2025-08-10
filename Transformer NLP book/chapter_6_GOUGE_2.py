# 导入所需的库
import pandas as pd
from datasets import load_dataset  # 用于加载CNN/DailyMail数据集
from collections import defaultdict  # 用于存储不同模型的摘要结果
import nltk  # 用于句子分割
from nltk.tokenize import sent_tokenize  # NLTK的句子分割工具
from transformers import pipeline, set_seed  # Hugging Face的pipeline用于模型推理，set_seed确保结果可复现

# 加载CNN/DailyMail数据集（版本3.0.0）
# 数据集包含新闻文章和对应的人工摘要，适合测试摘要模型
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

# 选择训练集中的第二篇文章（索引1），并截取前2000个字符作为样本
# 这样可以避免输入过长，适合模型处理
sample = dataset["train"][1]["article"][:2000]
print(sample)  # 打印样本文章内容，便于检查

# 创建一个defaultdict来存储不同模型生成的摘要
summaries = defaultdict()

# 下载NLTK的punkt_tab资源，用于句子分割
# punkt_tab是NLTK的预训练句子分割模型
nltk.download("punkt_tab")

# 定义基线摘要函数：提取文章前三句作为摘要
def three_sentence_summaries(text):
    sentences = sent_tokenize(text)[:3]  # 将文本分割成句子，取前三句
    return "\n".join(sentences)  # 用换行符连接句子，形成摘要

# 生成基线摘要并存储
summaries["baseline"] = three_sentence_summaries(sample)
print(summaries)  # 打印当前摘要字典，检查基线摘要

# 使用GPT-2模型（gpt2-xl）进行文本生成（非典型摘要任务）
set_seed(42)  # 设置随机种子，确保结果可复现
pipe = pipeline("text-generation", model="gpt2-xl")  # 加载GPT-2模型
gpt2_query = sample + "\nTL;DR:"  # 在文章后添加“TL;DR:”提示，引导生成简短总结
outputs = pipe(gpt2_query, max_length=512, clean_up_tokenization_spaces=True)  # 生成文本，限制最大长度
# 提取生成的文本中“TL;DR:”后的内容，并分割成句子
summaries["gpt2"] = "\n".join(sent_tokenize(outputs[0]["generated_text"][len(gpt2_query):]))
print(summaries["gpt2"])  # 打印GPT-2生成的摘要

# 使用T5模型（t5-large）进行摘要提取
# T5是一个通用文本生成模型，pipeline会自动添加“summarize:”前缀
pipe = pipeline("summarization", model="t5-large")  # 加载T5模型
outputs = pipe(sample)  # 对样本文章生成摘要
# 将T5生成的摘要分割成句子并存储
summaries["t5"] = "\n".join(sent_tokenize(outputs[0]["summary_text"]))

# 使用BART模型（facebook/bart-large-cnn）进行摘要提取
# BART是为摘要任务优化的模型，适合新闻文章
pipe = pipeline("summarization", model="facebook/bart-large-cnn")  # 加载BART模型
outputs = pipe(sample)  # 生成摘要
# 将BART生成的摘要分割成句子并存储
summaries["bart"] = "\n".join(sent_tokenize(outputs[0]["summary_text"]))

# 使用PEGASUS模型（google/pegasus-cnn_dailymail）进行摘要提取
# PEGASUS是为新闻摘要任务预训练的模型，擅长提取关键信息并改写
pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail")  # 加载PEGASUS模型
outputs = pipe(sample)  # 生成摘要
# 将PEGASUS生成的摘要中的特殊标记“<n>”替换为“.\n”，以便按句子分行
summaries["pegasus"] = outputs[0]["summary_text"].replace("<n>", ".\n")

# 输出所有结果
print("GROUND TRUTH")  # 打印数据集提供的人工摘要（参考标准）
print(dataset["train"][1]["highlights"])
print("")

# 循环打印每种方法的摘要
for model_name in summaries.keys():
    print(f"{model_name.upper()}:")  # 打印模型名称（大写）
    print(summaries[model_name])  # 打印对应摘要
    print("")  # 空行分隔

# GROUND TRUTH
# Mentally ill inmates in Miami are housed on the "forgotten floor"
# Judge Steven Leifman says most are there as a result of "avoidable felonies"
# While CNN tours facility, patient shouts: "I am the son of the president"
# Leifman says the system is unjust and he's fighting for change .
#
# BASELINE:
# Editor's note: In our Behind the Scenes series, CNN correspondents share their experiences in covering news and analyze the stories behind the events.
# Here, Soledad O'Brien takes users inside a jail where many of the inmates are mentally ill. An inmate housed on the "forgotten floor," where many mentally ill inmates are housed in Miami before trial.
# MIAMI, Florida (CNN) -- The ninth floor of the Miami-Dade pretrial detention facility is dubbed the "forgotten floor."
#
# GPT2:
#  The mentally ill aren't being properly treated and are being housed in an environment that is not comfortable or safe.
# The mentally ill are also more likely to be arrested and not charged with a crime.
#
# T5:
# mentally ill inmates are housed on the ninth floor of a florida jail .
# most face drug charges or charges of assaulting an officer .
# judge says arrests often result from confrontations with police .
# one-third of all people in Miami-dade county jails are mental ill .
#
# BART:
# Mentally ill inmates are housed on the "forgotten floor" of Miami-Dade jail.
# Most often, they face drug charges or charges of assaulting an officer.
# Judge Steven Leifman says the arrests often result from confrontations with police.
# He says about one-third of all people in the county jails are mentally ill.
#
# PEGASUS:
# Mentally ill inmates are housed on the "forgotten floor" of a Miami jail ..
# Judge Steven Leifman says most are charged with "avoidable felonies".
# Leifman says confrontations with police exacerbate mental illness ..
# Prisoners have no shoes, laces or mattresses .

import evaluate

rouge_compute = evaluate.load("rouge")
# evaluate.load("rouge") 默认计算以下 ROUGE 变种：
# ROUGE-1：基于一元词（unigram）的重叠，衡量单个词的匹配程度。
# ROUGE-2：基于二元词（bigram）的重叠，衡量短语结构的相似性。
# ROUGE-L：基于最长公共子序列（LCS），衡量句子结构的相似性。
# ROUGE-Lsum：基于按句子拆分的 ROUGE-L，特别适用于评估多句文本（如摘要），计算每个句子的 ROUGE-L 并取平均值。
result = {}
for model_name in summaries.keys():
    result[model_name] = rouge_compute.compute(
        predictions=[summaries[model_name]],  # 预测摘要
        references=[dataset["train"][1]["highlights"]],  # 参考摘要
    )
data = pd.DataFrame.from_dict(result, orient="index", columns=["rouge1", "rouge2", "rougeL", "rougeLsum"])
data = data.astype(float).round(4)
print(data)
#           rouge1  rouge2  rougeL  rougeLsum
# baseline  0.3651  0.1452  0.2063     0.2857
# gpt2      0.2558  0.0238  0.1628     0.2326
# t5        0.3830  0.1304  0.2553     0.3830
# bart      0.4752  0.2222  0.3168     0.4158
# pegasus   0.5057  0.3294  0.4368     0.4828