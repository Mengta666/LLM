# 导入必要的库
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import evaluate
import pandas as pd
from tqdm.auto import tqdm

# 加载 samsum 数据集，包含 train、validation 和 test 分割
dataset = load_dataset("knkarthick/samsum")
# 打印训练集的特征，确认数据集结构（包含 id、dialogue、summary 字段）
print(dataset["train"].features)
# 检查数据集中是否存在空值（None 或空字符串）
# 遍历训练集，打印 dialogue 或 summary 为 None 或空字符串的样本
for s in dataset["train"]:
    if s["dialogue"] == "" or s["summary"] == "" or s["summary"] is None or s["dialogue"] is None:
        print(s)
# 输出示例：{'id': '13828807', 'dialogue': None, 'summary': 'problem with visualization of the content'}
# 注意：此检查发现至少一个样本 dialogue 为 None，需要过滤
# 过滤掉 dialogue 或 summary 为 None 的样本
# 使用 dataset.filter 移除无效数据，确保后续 tokenization 和训练不会出错
dataset = dataset.filter(lambda x: x["dialogue"] is not None and x["summary"] is not None)
# 注意：此过滤仅移除 None，未处理空字符串或仅空格字符串，可能需进一步检查
# 定义模型检查点为 google/pegasus-cnn_dailymail（预训练的 Pegasus 模型，适合新闻摘要）
check_point = "google/pegasus-cnn_dailymail"
# 初始化 tokenizer，用于将文本转换为 token IDs
tokenizer = AutoTokenizer.from_pretrained(check_point)

# 定义批处理 tokenization 函数，准备模型输入
def tokenizer_batch(batch):
    # 对 dialogue 进行 tokenization，生成编码器输入
    # max_length=1024 是 Pegasus 的最大输入长度，truncation=True 截断超长序列
    # padding="longest" 将批次中的序列填充到最长长度
    dialog_tokens = tokenizer(batch["dialogue"], padding="max_length", truncation=True, max_length=1024)
    # 使用 as_target_tokenizer 上下文，确保 summary 的 tokenization 适合解码器
    # 解码器可能需要不同的 token 处理（如特定的 padding 或结束 token）
    with tokenizer.as_target_tokenizer():
        # 对 summary 进行 tokenization，生成训练目标（labels）
        # max_length=128 适合 samsum 数据集的摘要长度
        summary_tokens = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=128)
    # 返回模型需要的字段：
    # input_ids：dialogue 的 token IDs，输入到编码器
    # attention_mask：指示哪些 token 是有效内容（1）或 padding（0）
    # labels：summary 的 token IDs，作为解码器的训练目标
    # decoder_attention_mask：summary 的注意力掩码，指示有效 token
    return {
        "input_ids": dialog_tokens["input_ids"],
        "attention_mask": dialog_tokens["attention_mask"],
        "labels": summary_tokens["input_ids"],
        "decoder_attention_mask": summary_tokens["attention_mask"]
    }
# 应用 tokenization 函数，批处理数据集
# batched=True 表示按批次处理，减少内存占用
dataset = dataset.map(tokenizer_batch, batched=True)
# 设置数据集格式为 PyTorch 张量，仅保留模型需要的字段
# 移除原始字段（dialogue、summary、id）以节省内存
columns = ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]
dataset.set_format(type="torch", columns=columns)
# 注意：decoder_attention_mask 未包含在 columns 中，可能需要添加

# 微调 Pegasus 模型
# 设置计算设备（优先使用 GPU，否则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练 Pegasus 模型并移动到指定设备
model = AutoModelForSeq2SeqLM.from_pretrained(check_point).to(device)
# 初始化数据整理器，自动处理批次中的 padding 和 labels
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# 配置训练参数
trainer_args = TrainingArguments(
    output_dir="./chapter_6/pegasus_summarizer",  # 模型和检查点保存目录
    num_train_epochs=1,  # 训练 2 个 epoch
    per_device_train_batch_size=1,  # 每设备训练批量大小为 1（减小内存占用）
    per_device_eval_batch_size=1,  # 每设备验证批量大小为 1
    logging_dir="./chapter_6/pegasus_summarizer/logs",  # 日志保存目录
    logging_steps=10,  # 每 10 步记录一次日志
    log_level="info",  # 日志级别为 info
    eval_strategy="epoch",  # 每个 epoch 评估一次验证集
    save_strategy="epoch",  # 每个 epoch 保存一次检查点
    save_total_limit=1,  # 最多保存 2 个检查点，删除较旧的
    load_best_model_at_end=True,  # 训练结束时加载最佳模型（基于验证集损失）
    gradient_accumulation_steps=16,  # 梯度累积 16 步，有效批量大小为 1*16=16
    learning_rate=5e-5,  # 初始学习率
    lr_scheduler_type="linear",  # 线性学习率调度器
    weight_decay=0.01,  # 权重衰减，防止过拟合
    warmup_steps=500,  # 预热 500 步，逐渐增加学习率
)
# 初始化 Trainer，用于管理训练过程
trainer = Trainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)
# 开始训练模型
trainer.train()
trainer.save_model()
# {'eval_loss': 0.3107580244541168, 'eval_runtime': 98.0149, 'eval_samples_per_second': 8.346, 'eval_steps_per_second': 8.346, 'epoch': 1.0}

# 定义批处理生成器函数，按 batch_size 切分数据集
def get_batch_text(batch_size, dataset):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]
# 定义评估函数，计算模型在测试集上的 ROUGE 分数
def evaluate_summaris_pegasus(dataset, model, tokenizer, batch_size=8, device=device):
    # 将 dialogue 和 summary 分批
    dialogue_batches = list(get_batch_text(batch_size, dataset["dialogue"]))
    summary_batches = list(get_batch_text(batch_size, dataset["summary"]))
    # 初始化进度条，显示生成摘要的进度
    progress_bar = tqdm(range(len(dataset["dialogue"])), desc="Generating summaries in test dataset")
    gen_summaries = []  # 存储模型生成的摘要
    org_summaries = []  # 存储原始参考摘要
    # 逐批处理 dialogue 和 summary
    for dialogue, summary in zip(dialogue_batches, summary_batches):
        org_summaries.extend(summary)  # 添加参考摘要
        # 对 dialogue 进行 tokenization，生成模型输入
        input_ids = tokenizer(
            dialogue,
            max_length=1024,  # 与训练时一致，适合 Pegasus 的最大输入长度
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        # 使用模型生成摘要（无梯度计算以节省内存）
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                length_penalty=0.8,  # 控制生成长度，鼓励稍短的摘要
                num_beams=4,  # 使用 4 个 beam 搜索，提高生成质量
                max_length=128  # 生成摘要的最大长度
            )
        # 解码生成的 token IDs 为文本，移除特殊 token 并清理空格
        summaries_text = [
            (tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True)).replace("<n>", " ")
            for i in output
        ]
        gen_summaries.extend(summaries_text)  # 添加生成的摘要
        progress_bar.update(len(dialogue))  # 更新进度条
    # 加载 ROUGE 评估指标
    rouge_compute = evaluate.load("rouge")
    # 计算 ROUGE 分数，比较生成摘要和参考摘要
    return rouge_compute.compute(
        predictions=gen_summaries,
        references=org_summaries
    )

# 在测试集上评估微调后的模型
result = evaluate_summaris_pegasus(dataset["test"], trainer.model, tokenizer, batch_size=8)
# 将 ROUGE 分数格式化为 DataFrame 并打印
print(pd.DataFrame.from_dict(result, orient="index", columns=["test datasets in PEGASUS"]).T)
#                           rouge1    rouge2    rougeL     rougeLsum
# test datasets in PEGASUS  0.438458  0.209083  0.346912   0.347122