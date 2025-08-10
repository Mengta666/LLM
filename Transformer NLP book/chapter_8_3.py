# 量化

from transformers import pipeline

check_point = "./chapter_8/chapter_8_2/opt"
pipe = pipeline("text-classification", model=check_point)
state_dict = pipe.model.state_dict()
# print(state_dict.keys())
import matplotlib.pyplot as plt
weights = state_dict["distilbert.transformer.layer.0.attention.out_lin.weight"]
plt.hist(weights.cpu().numpy().flatten(), bins=250, range=(-0.3, 0.3), edgecolor="C0")
plt.show()

import torch
# 量化
zero_point = 0         #让f的零点依旧映射到q的零点上
# 这里将float32映射到8位整数
scale = (weights.max() - weights.min()) / (2 ** 8 - 1)
# 使用round()四舍五入最终计算的值，并将超过127或者低于-128的数强制转换为-128或者127
weight_q = (weights / scale).round().clamp(-128, 127).to(torch.int8)
plt.hist(weight_q.cpu().numpy().flatten(), bins=250, range=(-128, 127), edgecolor="C0")
plt.show()
print(weight_q)
# 上面可以用torch.quantize_per_tensor()一步解决
from torch import quantize_per_tensor
quantized_weights = quantize_per_tensor(weights, scale, zero_point, torch.qint8)

# 反量化，也就是恢复到float32，但是精度肯定会有所下降
weight_f = (weight_q * scale).to(torch.float32)
plt.hist(weight_f.cpu().numpy().flatten(), bins=250, range=(-0.3, 0.3), edgecolor="C0")
plt.show()
print(weight_f)

torch.save(quantized_weights, "quantized_weights.pt")
torch.save(weights, "weights.pt")

# 计算量化后的张量乘法速度快了多少(逐元素相乘)
import timeit
set_up = """
import torch
from torch.nn.quantized import QFunctional
Q_fuc = QFunctional()
quantized_weights = torch.load('quantized_weights.pt').to('cpu')
weights = torch.load('weights.pt').to('cpu')
"""
quantized_weights_stmt = """
Q_fuc.mul(quantized_weights,quantized_weights)
"""
weights_stmt = """
weights @ weights
"""

quantized_weights_time = timeit.timeit(stmt=quantized_weights_stmt, setup=set_up, number=1000)
weights_stmt_time = timeit.timeit(stmt=weights_stmt, setup=set_up, number=1000)
# Quantized weights multiplication time: 0.059184 seconds
# Original weights multiplication time: 1.311150 seconds
print(f"Quantized weights multiplication time: {quantized_weights_time:.6f} seconds")
print(f"Original weights multiplication time: {weights_stmt_time:.6f} seconds")
import os
os.remove("quantized_weights.pt")
os.remove("weights.pt")


# 现在对前面的蒸馏模型进行动态量化
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(check_point).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(check_point)
from torch.quantization import quantize_dynamic
model_quantize = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model.config.save_pretrained("./chapter_8/chapter_8_2/quantize_model")
tokenizer.save_pretrained("./chapter_8/chapter_8_2/quantize_model")
torch.save(model_quantize.state_dict(), "./chapter_8/chapter_8_2/quantize_model/quantized_model.pt")

pipe = pipeline("text-classification", model=model_quantize, tokenizer=tokenizer, device="cpu")
from chapter_8_1 import PerformanceBenchmark
from datasets import load_dataset
dataset = load_dataset("clinc_oos", "plus")
benchmark = PerformanceBenchmark(pipe, dataset["test"], optim_type="dynamic_quantization")
# Model size: 132.40 MB
quantize_result = benchmark.run()

pipe = pipeline("text-classification", model=check_point, device="cpu")
benchmark = PerformanceBenchmark(pipe, dataset["test"], optim_type="opt_model")
# Model size: 255.89 MB
opt_result = benchmark.run()

pipe = pipeline("text-classification", model="transformersbook/bert-base-uncased-finetuned-clinc", device="cpu")
benchmark = PerformanceBenchmark(pipe, dataset["test"], optim_type="model_baseline")
# Model size: 418.16 MB
baseline_result = benchmark.run()

import matplotlib.pyplot as plt

plt.ylim(80, 95)
plt.xlim(quantize_result["dynamic_quantization"]["time"]["time_avg"] - 3, baseline_result["model_baseline"]["time"]["time_avg"] + 3)
plt.scatter(
    quantize_result["dynamic_quantization"]["time"]["time_avg"],
    quantize_result["dynamic_quantization"]["accuracy"]["accuracy"] * 100,
    label="opt_quantize_result"
)
plt.scatter(
    opt_result["opt_model"]["time"]["time_avg"],
    opt_result["opt_model"]["accuracy"]["accuracy"] * 100,
    label="opt_result"
)
plt.scatter(
    baseline_result["model_baseline"]["time"]["time_avg"],
    baseline_result["model_baseline"]["accuracy"]["accuracy"] * 100,
    label="baseline_result"
)
plt.xlabel("Time (s)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()





