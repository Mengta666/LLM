# 知识蒸馏
from typing import Union, Any, Optional
import torch
from torch import nn
import numpy as np
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# 定义知识蒸馏的训练参数类，继承自TrainingArguments
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        # 调用父类初始化方法
        super().__init__(*args, **kwargs)
        # alpha：学生模型损失和蒸馏损失的权重系数，0.5表示两者各占一半
        self.alpha = alpha
        # temperature：温度参数，用于软化logits的分布，控制蒸馏的平滑程度
        self.temperature = temperature

# 定义知识蒸馏的训练器类，继承自Trainer
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model, **kwargs):
        # 调用父类初始化方法
        super().__init__(*args, **kwargs)
        # 保存教师模型，用于计算蒸馏损失
        self.teacher_model = teacher_model

    def compute_loss(
        self,
        model: nn.Module,  # 学生模型
        inputs: dict[str, Union[torch.Tensor, Any]],  # 输入数据（如input_ids, attention_mask等）
        return_outputs: bool = False,  # 是否返回学生模型的输出
        num_items_in_batch: Optional[torch.Tensor] = None,  # 批次大小（可选）
    ):
        # 计算学生模型的输出
        outputs_stu = model(**inputs)
        logits_stu = outputs_stu.logits  # 学生模型的logits
        loss_stu = outputs_stu.loss  # 学生模型的原始损失（如交叉熵损失）

        # 使用no_grad以避免计算教师模型的梯度，节省内存
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits  # 教师模型的logits

        # 定义KL散度损失函数，reduction="batchmean"表示按批次平均
        loss_fc = nn.KLDivLoss(reduction="batchmean")
        # 计算知识蒸馏损失
        # 学生logits除以温度后进行log_softmax，教师logits除以温度后进行softmax
        # 温度平方用于缩放KL散度损失，保持量级一致
        loss_kd = self.args.temperature ** 2 * loss_fc(
            nn.functional.log_softmax(logits_stu / self.args.temperature, dim=-1),
            nn.functional.softmax(logits_tea / self.args.temperature, dim=-1)
        )
        # 总损失 = alpha * 学生损失 + (1 - alpha) * 蒸馏损失
        loss = self.args.alpha * loss_stu + (1 - self.args.alpha) * loss_kd
        # 根据return_outputs决定返回损失值还是(损失值, 学生输出)
        return (loss, outputs_stu) if return_outputs else loss

# 加载clinc_oos数据集（plus版本），用于意图分类任务
dataset = load_dataset("clinc_oos", "plus")
# 定义教师模型和学生模型的checkpoint
tea_checkpoint = "transformersbook/bert-base-uncased-finetuned-clinc"
stu_checkpoint = "distilbert-base-uncased"

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
# 加载学生模型的分词器
stu_tokenizer = AutoTokenizer.from_pretrained(stu_checkpoint)
# 定义分词函数，对数据集中的文本进行分词和截断
def tokenize_function(examples):
    return stu_tokenizer(examples["text"], truncation=True)
# 对数据集进行分词处理，并移除原始文本列
clinc_enc = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# 将意图列重命名为labels，以便与Trainer兼容
clinc_enc = clinc_enc.rename_column("intent", "labels")

# 定义评估指标函数，计算准确率
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # 取logits最大值的索引作为预测类别
    import evaluate
    compute_accuracy = evaluate.load("accuracy")
    return compute_accuracy.compute(predictions=predictions, references=labels)

from transformers import DataCollatorWithPadding
# 定义数据整理器，自动对批次数据进行填充
data_collator = DataCollatorWithPadding(tokenizer=stu_tokenizer)
# 定义训练参数
batch_size = 48
args = DistillationTrainingArguments(
    output_dir=r".\chapter_8\chapter_8_2",  # 输出目录
    eval_strategy="epoch",  # 每个epoch进行一次评估
    num_train_epochs=5,  # 训练5个epoch
    lr_scheduler_type="linear",  # 使用线性学习率调度器
    learning_rate=2e-5,  # 初始学习率
    per_device_train_batch_size=48,  # 每个设备的训练批次大小
    per_device_eval_batch_size=48,  # 每个设备的评估批次大小
    weight_decay=0.01,  # 权重衰减，防止过拟合
    warmup_steps=500,  # 学习率预热步数
    save_strategy="epoch",  # 每个epoch保存一次模型
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
    alpha=0.5,  # 学生损失和蒸馏损失的权重
    temperature=2.0,  # 蒸馏温度
)

from transformers import pipeline
# 加载教师模型的pipeline，用于获取标签映射
pipe = pipeline("text-classification", model="transformersbook/bert-base-uncased-finetuned-clinc")
int2label = pipe.model.config.id2label  # 类别ID到标签的映射
label2int = pipe.model.config.label2id  # 标签到类别ID的映射
intents = dataset["train"].features["intent"]
# 配置学生模型，设置类别数和标签映射
stu_config = AutoConfig.from_pretrained(stu_checkpoint, num_labels=intents.num_classes, id2label=int2label, label2id=label2int)

# 检测可用设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义学生模型初始化函数
def stu_model_init():
    return AutoModelForSequenceClassification.from_pretrained(stu_checkpoint, config=stu_config).to(device)
# 加载教师模型
tea_model = AutoModelForSequenceClassification.from_pretrained(tea_checkpoint, num_labels=intents.num_classes).to(device)

# 初始化知识蒸馏训练器
trainer = DistillationTrainer(
    model_init=stu_model_init,  # 学生模型初始化函数
    teacher_model=tea_model,  # 教师模型
    args=args,  # 训练参数
    train_dataset=clinc_enc["train"],  # 训练数据集
    eval_dataset=clinc_enc["validation"],  # 验证数据集
    tokenizer=stu_tokenizer,  # 分词器
    data_collator=data_collator,  # 数据整理器
    compute_metrics=compute_metrics,  # 评估指标
)
# 开始训练
trainer.train()
# 保存训练好的模型
trainer.save_model()
# 导入自定义绘图函数并可视化结果
from chapter_8_2_plot import plot_results
plot_results(dataset)

# # 找到最优的超参数 alpha 和 temperature
# # 方法1：网格搜索
# alpha_list = np.linspace(0.0, 1.0, 20).tolist()  # alpha从0到1，均匀取20个值
# T_list = np.linspace(1.0, 10.0, 20).tolist()  # temperature从1到10，均匀取20个值
# # 存储每次实验的结果
# results = []
# for alpha in alpha_list:
#     for T in T_list:
#         # 为每次实验创建唯一的输出目录
#         output_dir = f"./chapter_8/chapter_8_2/alpha_{alpha:.2f}_T_{T:.2f}"
#         # 更新训练参数
#         args = DistillationTrainingArguments(
#             output_dir=output_dir,
#             eval_strategy="epoch",
#             num_train_epochs=5,
#             lr_scheduler_type="linear",
#             learning_rate=2e-5,
#             per_device_train_batch_size=48,
#             per_device_eval_batch_size=48,
#             weight_decay=0.01,
#             warmup_steps=500,
#             save_strategy="epoch",
#             load_best_model_at_end=True,
#             alpha=alpha,
#             temperature=T,
#         )
#         # 重新初始化学生模型，避免权重累积
#         stu_model = AutoModelForSequenceClassification.from_pretrained(stu_checkpoint, config=stu_config).to(device)
#         # 创建新的训练器实例
#         trainer = DistillationTrainer(
#             model=stu_model,
#             teacher_model=tea_model,
#             args=args,
#             train_dataset=clinc_enc["train"],
#             eval_dataset=clinc_enc["validation"],
#             tokenizer=stu_tokenizer,
#             data_collator=data_collator,
#             compute_metrics=compute_metrics,
#         )
#         # 训练模型
#         trainer.train()
#         # 保存模型
#         trainer.save_model()
#         # 获取评估结果
#         eval_results = trainer.evaluate()
#         results.append({
#             'alpha': alpha,
#             'temperature': T,
#             'accuracy': eval_results.get('eval_accuracy', 0.0)
#         })
# # 打印所有实验结果
# for result in results:
#     print(f"Alpha: {result['alpha']:.2f}, Temperature: {result['temperature']:.2f}, Accuracy: {result['accuracy']:.4f}")
# # 找到最佳超参数组合
# best_result = max(results, key=lambda x: x['accuracy'])
# print(
#     f"Best Alpha: {best_result['alpha']:.2f}, Best Temperature: {best_result['temperature']:.2f}, Best Accuracy: {best_result['accuracy']:.4f}")
# # 可视化网格搜索结果
# # plot_results(dataset)

# # 方法2：随机网格搜索
# np.random.seed(42)  # 设置随机种子以确保结果可重复
# alpha_list = np.random.uniform(0.1, 1.0, 20).tolist()  # 在[0.1, 1.0]随机取20个alpha值
# temperature_list = np.random.uniform(1, 10, 20).tolist()  # 在[1, 10]随机取20个temperature值
# # 循环逻辑同方法1

# 方法3：贝叶斯优化搜索
# 使用Optuna库进行超参数优化
import optuna
# 定义超参数搜索空间
def hp_space(trial):
    return {
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10),  # epoch数在5到10之间
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),  # alpha在[0, 1]
        "temperature": trial.suggest_float("temperature", 2, 20),  # temperature在[2, 20]
    }
# 执行超参数搜索，优化目标为最大化准确率
best_run = trainer.hyperparameter_search(n_trials=20, direction="maximize", hp_space=hp_space)
# 打印最佳超参数和结果
# BestRun(run_id='14', objective=0.9483870967741935, hyperparameters={'num_train_epochs': 10, 'alpha': 0.5796247473161299, 'temperature': 19.558884820628087}, run_summary=None)
print(best_run)
# 将最佳超参数应用到训练参数中
for k, v in best_run.hyperparameters.items():
    setattr(args, k, v)
# 使用最佳超参数重新初始化训练器
trainer_opt = DistillationTrainer(
    model_init=stu_model_init,
    teacher_model=tea_model,
    args=args,
    train_dataset=clinc_enc["train"],
    eval_dataset=clinc_enc["validation"],
    tokenizer=stu_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# 使用最佳超参数进行训练
trainer_opt.train()
# 保存优化后的模型
trainer_opt.save_model("./chapter_8/chapter_8_2/opt")
# 可视化优化结果
plot_results(dataset, "chapter_8/chapter_8_2/opt")