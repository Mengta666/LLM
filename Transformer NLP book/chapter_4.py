# 导入必要的库
from collections import defaultdict  # 用于创建默认字典，自动初始化不存在的键
from datasets import DatasetDict, load_dataset  # DatasetDict 用于存储数据集，load_dataset 用于加载数据集
import pandas as pd
from transformers import AutoTokenizer  # 用于加载预训练分词器
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput  # 配置和输出类，用于自定义模型
from transformers.models.roberta.modeling_roberta import (RobertaModel, RobertaPreTrainedModel)  # RoBERTa 模型基类
import torch.nn

# 设置显示所有行
pd.set_option('display.max_rows', None)
# 设置显示所有列
pd.set_option('display.max_columns', None)
# 设置列宽，防止内容被截断
pd.set_option('display.max_colwidth', None)
# 设置显示宽度，防止换行
pd.set_option('display.width', None)

# 定义语言列表和对应的采样比例
langs = ["de", "fr", "it", "en"]  # 要处理的语言：德语、法语、意大利语、英语
fracs = [0.629, 0.229, 0.084, 0.059]  # 每种语言的采样比例，用于从数据集中抽取部分样本

# 创建一个默认字典，键为语言代码，值为 DatasetDict 对象
# defaultdict 确保访问不存在的键时自动创建 DatasetDict 实例，简化初始化
panx_ch = defaultdict(DatasetDict)

# 加载和处理数据集
for lang, frac in zip(langs, fracs):
    # 加载 xtreme 数据集的 PAN-X 子集，指定语言（如 PAN-X.de）
    ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
    for split in ds:  # 遍历数据集的每个拆分（如 train、validation、test）
        # 对每个拆分进行随机打乱（固定种子以保证可重复性），并按比例采样
        panx_ch[lang][split] = ds[split].shuffle(seed=0).select(range(int(frac * ds[split].num_rows)))

# 使用 pandas 显示每个语言的训练样本数量
# 创建 DataFrame，列名为语言，行索引为 "Number of training examples"
print(pd.DataFrame({lang: [panx_ch[lang]["train"].num_rows] for lang in langs},
                   index=["Number of training examples"]))

# 查看德语训练集的第一个样本
# 数据集已预处理（按空格分词，包含 tokens 和 ner_tags）
element = panx_ch["de"]["train"][0]
print(element)
# 输出示例：{'tokens': ['2.000', 'Einwohnern', ...], 'ner_tags': [0, 0, ...], ...}

# 获取 NER 标签的定义
# tag 是 ClassLabel 对象，包含标签名称（如 ['O', 'B-PER', 'I-PER', ...]）
tags = panx_ch["de"]["train"].features["ner_tags"].feature
print(tags.names)  # 输出：['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# 定义函数将数字标签转换为字符串标签
def id2str(example):
    # 将 ner_tags 的数字索引转换为对应的字符串标签（如 0 -> 'O', 5 -> 'B-LOC'）
    return {"ner_tags_str": [tags.int2str(idx) for idx in example["ner_tags"]]}

# 对每种语言的数据集应用 id2str 函数，添加 ner_tags_str 字段
for lang in langs:
    panx_ch[lang] = panx_ch[lang].map(id2str, batched=True)

# 显示德语训练集第一个样本的 tokens 和 ner_tags_str
# 使用 pandas DataFrame 直观展示 tokens 和对应的 NER 标签
print(pd.DataFrame([panx_ch["de"]["train"][0]["tokens"], panx_ch["de"]["train"][0]["ner_tags_str"]],
                   index=["Tokens", "Tags"]))
# 输出示例：
#            0           1   2    3   ...          8             9        10 11
# Tokens  2.000  Einwohnern  an  der  ...  polnischen  Woiwodschaft  Pommern  .
# Tags        O           O   O    O  ...       B-LOC         B-LOC    I-LOC  O

# 提取德语数据集
panx_de = panx_ch["de"]
print("数据处理完毕")

# 加载分词器
bert_model_name = "bert-base-cased"  # BERT 模型，使用 WordPiece 分词算法
xlmr_model_name = "xlm-roberta-base"  # XLM-RoBERTa 模型，使用 SentencePiece BPE 分词算法
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

# 示例：对德语训练集第一个样本的 tokens 进行分词
text = element["tokens"]  # 原始 tokens，如 ['2.000', 'Einwohnern', ...]
# 使用两个分词器处理，is_split_into_words=True 表示输入已分词
print(pd.DataFrame([bert_tokenizer(text, is_split_into_words=True).tokens(),
                    xlmr_tokenizer(text, is_split_into_words=True).tokens()],
                   index=["BERT", "XLM-R"]))
# 输出示例：
#           0       1           2    3    4   ...    27      28    29    30     31
# BERT   [CLS]       2           .  000  Ein  ...    Po  ##mmer   ##n     .  [SEP]
# XLM-R    <s>  ▁2.000  ▁Einwohner    n  ▁an  ...  None    None  None  None   None

# 定义分词和标签对齐函数
def tokenize_and_align_labels(batch):
    # 使用 XLM-RoBERTa 分词器处理 tokens，truncation=True 确保序列不超过最大长度
    # is_split_into_words=True 表示输入是预分词的单词列表
    token_inputs = xlmr_tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for idx, label in enumerate(batch["ner_tags"]):
        # 获取子词到原始单词的映射索引
        word_ids = token_inputs.word_ids(batch_index=idx)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                # 特殊标记（如 <s>, </s>）或子词延续（如 'schaft'）标记为 -100
                # -100 是 PyTorch CrossEntropyLoss 的 ignore_index，忽略损失计算
                label_ids.append(-100)
            else:
                # 为每个原始单词的首个子词分配对应的 NER 标签
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    token_inputs["labels"] = labels  # 添加对齐后的标签
    return token_inputs

# 编码数据集
def encode_panx_dataset(corpus):
    # 应用 tokenize_and_align_labels，移除原始字段，保留分词结果
    corpus = corpus.map(tokenize_and_align_labels, batched=True,
                        remove_columns=["langs", "ner_tags", "tokens", "ner_tags_str"])
    return corpus

# 对德语数据集进行分词和标签对齐
panx_de = encode_panx_dataset(panx_de)
# 查看第一个样本的处理结果
print(panx_de["train"][0])
# 输出示例：
# {
#     'ner_tags_str': ['O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'B-LOC', 'B-LOC', 'I-LOC', 'O'],
#     'input_ids': [0, 70101, 176581, 19, 142, 122, 2290, 708, 1505, 18363, 18, 23, 122, 127474, 15439, 13787, 14, 15263, 18917, 663, 6947, 19, 6, 5, 2],
#     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     'labels': [-100, 0, 0, -100, 0, 0, 5, -100, -100, 6, -100, 0, 0, 5, -100, 5, -100, -100, -100, 6, -100, -100, 0, -100, -100]
# }
print("数据集分词处理完毕")

# 自定义 XLM-RoBERTa 模型用于 token 分类（NER）
class XLMRobertaForClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig  # 使用 XLM-RoBERTa 的配置类

    def __init__(self, config):
        super().__init__(config)  # 初始化父类（RobertaPreTrainedModel）
        self.num_labels = config.num_labels  # 标签数量（如 7：O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC）
        # 定义 RoBERTa 主干模型，不添加池化层（NER 不需要句子级别表示）
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 添加 Dropout 层，用于正则化，减少过拟合
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # 定义分类器，将隐藏状态映射到标签空间
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # 初始化权重（继承自父类的权重初始化方法）
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # 前向传播：通过 RoBERTa 主干模型获取隐藏状态
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # 提取序列表示（每个 token 的隐藏状态）
        # 应用 Dropout 进行正则化
        sequence_output = self.dropout(sequence_output)
        # 通过线性层生成每个 token 的分类 logits（未归一化的概率）
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            # 如果提供了标签，计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss()
            # 将 logits 和 labels 展平，计算损失（-100 的位置被忽略）
            # logits 形状为 (batch_size, seq_length, 7)，每个 token 有 7 个标签的得分。
            # -1 表示将前两个维度（batch_size 和 seq_length）合并为一个维度（总样本数，展平）。
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # 返回 TokenClassifierOutput，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 加载自定义模型
index2tag = {idx:tag for idx, tag in enumerate(tags.names)}
tag2index = {tag:idx for idx, tag in enumerate(tags.names)}

from transformers import AutoConfig
xlmr_config = AutoConfig.from_pretrained(xlmr_model_name, num_labels=tags.num_classes,
                                         idx2label=index2tag, label2idx=tag2index)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlmr_model = XLMRobertaForClassification.from_pretrained(xlmr_model_name, config=xlmr_config).to(device)
print("模型加载完毕")
# 定义一个预测函数
def tag_text(text, tags, model, tokenizer):
    tokens = tokenizer(text).tokens()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)[0]
    predictions = torch.argmax(outputs, dim=2)
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens, preds], index=["tokens", "predictions"])

#                  0      1      2      3  ...      6      7      8      9
# tokens         <s>  ▁Jack  ▁Spar    row  ...   ▁New  ▁York      !   </s>
# predictions  B-PER  B-PER  B-LOC  B-PER  ...  B-PER  B-PER  B-PER  B-PER
# 不微调结果很不理想
print(tag_text("Jack Sparrow loves New York!", tags, xlmr_model, xlmr_tokenizer))

# 进行微调
print("开始微调")
import numpy as np
from sklearn.metrics import f1_score
def compute_metrics(pred):
    labels = pred.label_ids  # 形状: (batch_size, seq_length)
    predictions = pred.predictions  # 形状: (batch_size, seq_length, num_labels)
    preds = np.argmax(predictions, axis=2)  # 形状: (batch_size, seq_length)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        for idx in range(seq_len):
            if labels[batch_idx][idx] != -100:  # 忽略 -100 的位置
                labels_list.append(index2tag[labels[batch_idx][idx]])
                preds_list.append(index2tag[preds[batch_idx][idx]])
    f1 = f1_score(labels_list, preds_list, average="weighted")
    return {
        "accuracy": (np.array(preds_list) == np.array(labels_list)).mean(),
        "f1": f1,
    }
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
from transformers import Trainer, TrainingArguments
training = TrainingArguments(
    output_dir = "chapter_4_de",
    logging_dir = "chapter_4_de/logs",
    log_level="info",
    logging_steps=100,  # 每 100 步记录一次日志
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",  # 每轮保存模型
    weight_decay=0.01,
    learning_rate=5e-5,
    warmup_steps=500,  # 学习率预热步数
    lr_scheduler_type="linear",  # 线性学习率调度
    load_best_model_at_end=True,
    disable_tqdm=False,
    metric_for_best_model="f1",
)
trainer = Trainer(
    model=xlmr_model,
    args=training,
    train_dataset=panx_de["train"],
    eval_dataset=panx_de["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("chapter_4/best_model_de")
#               0      1      2      3   ...   10          11     12    13
# tokens       <s>  ▁Jeff    ▁De     an  ...  ▁in  ▁Kaliforni     en  </s>
# predictions    O  B-PER  I-PER  I-PER  ...    O       B-LOC  I-LOC     O
print(tag_text("Jeff Dean ist ein Informatiker bei Google in Kalifornien", tags, trainer.model, xlmr_tokenizer))
print("微调完成")

# 错误分析
# 对于一些预测错误的的结果（或者说高损失的结果），查看哪些被错误分类
from torch.nn.functional import cross_entropy
def forword_pass_with_label(batch):
    # 将批量数据（batch）转换为一个字典列表，其中每个字典表示一个样本，键是数据集的字段名，值是对应样本的数据。
    # batch = {
    #     'input_ids': [[101, 102, ...], [101, 103, ...], ...],
    #     'attention_mask': [[1, 1, ...], [1, 1, ...], ...],
    #     'labels': [[0, -100, ...], [0, -100, ...], ...]}
    # 使用 * 解包操作符，将 batch.values() 的内容解包为 zip 函数的参数。每个字段的值是一个列表，zip(*batch.values()) 会将这些列表按索引（样本）对齐，生成一个元组的迭代器。
    # [
    #     ([101, 102, ...], [1, 1, ...], [0, -100, ...]),  # 第 1 个样本的所有字段值
    #     ([101, 103, ...], [1, 1, ...], [0, -100, ...]),  # 第 2 个样本的所有字段值
    #     ...
    # ]
    # 最终 features 是一个列表，列表中的每个元素是一个字典，表示一个样本的所有字段和值。
    features = [dict(zip(batch.keys(), t)) for t in zip(*batch.values())]
    # 填充为统一长度，并且填充的分类索引为 -100
    batch = data_collator(features)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    with torch.no_grad():
        outputs = trainer.model(input_ids, attention_mask=attention_mask)
        predicted = torch.argmax(outputs.logits, dim=2).cpu().numpy()
    loss = cross_entropy(outputs.logits.view(-1, tags.num_classes), labels.view(-1), reduction="none")
    # 将一维 loss 张量（形状 batch_size * seq_length) 重塑为二维 (batch_size, seq_length)。
    loss = loss.view(len(input_ids), -1).cpu().numpy()
    return {
        "loss": loss,
        "predicted_label": predicted,
    }

valid_set = panx_de["validation"]
valid_set = valid_set.map(forword_pass_with_label, batched=True, batch_size=32)
df = valid_set.to_pandas()

# 处理-100的标签分类
index2tag[-100] = "PAD"
df["input_tokens"] = df["input_ids"].apply(lambda x:  xlmr_tokenizer.convert_ids_to_tokens(x))
df["predicted_label"] = df["predicted_label"].apply(lambda x: [index2tag[i] for i in x])
df["labels"] = df["labels"].apply(lambda x: [index2tag[i] for i in x])
# axis=1 按行操作
df["loss"] = df.apply(lambda x: x["loss"][:len(x["input_ids"])], axis=1)
df["predicted_label"] = df.apply(lambda x: x["predicted_label"][:len(x["input_ids"])], axis=1)
print("第一个预测与输出：")
# input_ids:  [0, 10699, 11, 15, 16104, 1388, 2]
# attention_mask: [1, 1, 1, 1, 1, 1, 1]
# labels: [PAD, B-ORG, PAD, I-ORG, I-ORG, I-ORG, PAD]
# loss: [0.0, 0.0018861376, 0.0, 0.0016913408, 0.0017037175, 0.0013633014, 0.0]
# predicted_label: [I-ORG, B-ORG, I-ORG, I-ORG, I-ORG, I-ORG, I-ORG]
# input_tokens: [<s>, ▁Ham, a, ▁(, ▁Unternehmen, ▁), </s>]
print(df.head(1))

# 使用 pandas.Series.explode() 方法将列表展开为多行（也就是每个元素对应的所有项）
df_tokens = df.explode(['input_ids', 'input_tokens', 'attention_mask', 'labels', 'loss', 'predicted_label'])
df_tokens = df_tokens.query("labels != 'PAD'")
# 保留两位小数
df_tokens["loss"] = df_tokens["loss"].astype(float).round(2)
print("输入的ID、输入的token、注意力掩码、标签、损失和预测标签（第一个）：")
#                      0      0             0      0      1      1      1
# input_ids        10699     15         16104   1388  56530  83982     10
# attention_mask       1      1             1      1      1      1      1
# labels           B-ORG  I-ORG         I-ORG  I-ORG      O  B-ORG  I-ORG
# loss               0.0    0.0           0.0    0.0    0.0    1.8   2.01
# predicted_label  B-ORG  I-ORG         I-ORG  I-ORG      O  B-LOC  I-LOC
# input_tokens      ▁Ham     ▁(  ▁Unternehmen     ▁)    ▁WE   ▁Luz     ▁a
print(df_tokens.head(7).T)
# 对于每一个标签对应的损失进行汇总，得到损失最大的标签
print("最大的标签和损失：")
#               0       1        2        3        4       5       6
# labels    I-ORG       O    B-ORG    B-LOC    I-LOC   I-PER   B-PER
# mean       0.56    0.04     0.68     0.38     0.73    0.23    0.33
# count      3820   43648     2683     3172     1462    4139    2893
# sum     2127.61  1931.7  1835.42  1190.97  1071.54  954.75  941.48
print((df_tokens.groupby("labels")[["loss"]].agg(["mean", "count", "sum"]).droplevel(level=0, axis=1)
 .sort_values(by="sum", ascending=False).reset_index().round(2).T))
# 计算每一个句子的总损失，并从大到小排序，取前 10 个
df["total_loss"] = df["loss"].apply(sum)
df_tmp = df.sort_values(by="total_loss", ascending=False).head(3)
def get_samples(df_tmp):
    for _, row in df_tmp.iterrows():
        labels, preds, tokens, losses = [], [], [], []
        for i, mask in enumerate(row["attention_mask"]):
            """不要为0的部分"""
            if mask not in {0, len(row["attention_mask"])}:
                labels.append(row["labels"][i])
                preds.append(row["predicted_label"][i])
                tokens.append(row["input_tokens"][i])
                losses.append(row["loss"][i])
        df_tmp = pd.DataFrame({"labels": labels, "preds": preds, "tokens": tokens, "losses": losses}).T
        yield df_tmp
print("错误样本：")
#            0         1         2         3            4         5      6         7        8         9         10        11        12        13         14     15
# labels    PAD     B-PER     I-PER     I-PER          PAD     I-PER    PAD     I-PER      PAD     I-PER     I-PER     I-PER     I-PER     I-PER      I-PER    PAD
# preds   I-ORG     B-ORG     I-ORG     I-ORG        I-ORG     I-ORG  I-ORG     I-ORG    I-ORG     I-ORG     I-ORG     I-ORG     I-ORG     I-ORG      I-ORG  I-ORG
# tokens    <s>   ▁United  ▁Nations    ▁Multi  dimensional  ▁Integra    ted   ▁Stabil  ization  ▁Mission       ▁in      ▁the  ▁Central  ▁African  ▁Republic   </s>
# losses    0.0  8.097714  7.505383  7.671896          0.0  7.495738    0.0  7.271898      0.0   7.46447  7.173132  7.507694  7.637827  7.509881   7.130208    0.0
#            0         1    2    3         4          5          6         7     8         9         10        11        12          13     14      15        16     17     18
# labels    PAD     B-ORG  PAD  PAD     I-ORG      I-ORG      I-ORG     I-ORG   PAD     I-ORG     I-ORG     I-ORG     I-ORG       I-ORG    PAD     PAD     I-ORG    PAD    PAD
# preds   I-ORG         O    O    O         O          O          O         O     O         O         O         O         O       B-ORG  I-ORG   I-ORG     I-ORG  I-ORG  I-ORG
# tokens    <s>       ▁''    8    .     ▁Juli        ▁''         ▁:  ▁Protest  camp      ▁auf      ▁dem  ▁Gelände      ▁der  ▁Republika      n  ischen      ▁Gar     de   </s>
# losses    0.0  9.253578  0.0  0.0  7.628485  10.918892  10.151277  6.322905   0.0  7.166994  9.577691  7.268137  7.307208     6.47056    0.0     0.0  0.002527    0.0    0.0
#          0         1         2        3    4         5         6          7          8         9         10        11     12        13        14   15        16        17   18    19
# labels  PAD         O         O        O  PAD         O         O      B-LOC      I-LOC     I-LOC     I-LOC     I-LOC    PAD     I-LOC     I-LOC  PAD     I-LOC     I-LOC  PAD   PAD
# preds     O         O         O    B-ORG    O         O         O          O          O     B-ORG         O         O  I-ORG         O         O    O         O         O    O     O
# tokens  <s>        ▁'       ▁''       ▁Τ    Κ       ▁''        ▁'         ▁'        ▁''        ▁T       ▁''        ▁'     ri       ▁''        ▁'    k       ▁''        ▁'  ala  </s>
# losses  0.0  0.000057  0.000081  0.77541  0.0  0.000151  0.000062  12.315774  11.603943  9.314952  6.544359  6.597558    0.0  7.010038  7.169928  0.0  7.600509  7.980705  0.0   0.0
for sample in get_samples(df_tmp):
    print(sample)
print("错误分析完成")

# 多语言模型微调策略
# 前面只是对单一语言进行微调
# 整个单一微调模型的在其他语言上的效果：
# metrics的参数：
# {'test_loss': 0.162, 'test_accuracy': 0.965, 'test_f1': 0.965, 'test_runtime': 7.698,
# 'test_samples_per_second': 817.005, 'test_steps_per_second': 51.176}
print(f"metrics的参数：{trainer.predict(panx_de["test"]).metrics}")
def get_f1_score(lang, trainer):
    panx_ds = encode_panx_dataset(panx_ch[lang])
    return trainer.predict(panx_ds["test"]).metrics["test_f1"]
# F1-score of [de] model on [de] test set:0.9652037563020435
# F1-score of [de] model on [fr] test set:0.833460641964333
# F1-score of [de] model on [it] test set:0.829231166815888
# F1-score of [de] model on [en] test set:0.8083032193253991
for lang in langs:
    print(f"F1-score of [de] model on [{lang}] test set:{get_f1_score(lang, trainer)}")
# 可见其在其他零样本迁移语言上的效果并不是很好
# 进行多语言协同微调
# 将多种语言进行融合，并开始训练
# 使用hugging face中的concatenate_datasets()函数进行多个数据集组合: 所有输入数据集必须具有相同的特征（features），即相同的列名和列类型（例如，input_ids, attention_mask, labels 等）
from datasets import concatenate_datasets
# 组合德语和法语
panx_fr = encode_panx_dataset(panx_ch["fr"])
panx_de_fr_train = concatenate_datasets([panx_de["train"], panx_fr["train"]]).shuffle(seed=42)
panx_de_fr_eval = concatenate_datasets([panx_de["validation"], panx_fr["validation"]]).shuffle(seed=42)
panx_de_fr_test = concatenate_datasets([panx_de["test"], panx_fr["test"]]).shuffle(seed=42)
training.output_dir = "chapter_4_de_fr"
training.logging_dir = "chapter_4_de_fr/logs"
trainer_de_fr = Trainer(
    model=xlmr_model,
    args=training,
    train_dataset=panx_de_fr_train,
    eval_dataset=panx_de_fr_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer_de_fr.train()
trainer_de_fr.save_model("chapter_4/best_model_de_fr")
# F1-score of [de_fr] model on [de] test set:0.9661370702218528
# F1-score of [de_fr] model on [fr] test set:0.9305731509283232
# F1-score of [de_fr] model on [it] test set:0.9200392967041763
# F1-score of [de_fr] model on [en] test set:0.856609987636176
for lang in langs:
    print(f"F1-score of [de_fr] model on [{lang}] test set:{get_f1_score(lang, trainer_de_fr)}")
print("多语言协同微调完成")







