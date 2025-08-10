# 使用ONNX进行推理优化

# 设置高并发环境（对于cpu而言），下面使用在cpu上进行量化等优化
import os

import numpy as np
from psutil import cpu_count
os.environ["OMP_NUM_THREADS"] = str(cpu_count())
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
# 将蒸馏后的模型转换为onnx格式
check_point = "./chapter_8/chapter_8_2/opt"
onnx_output_dir = "./chapter_8/chapter_8_2/onnx"
ort_model = ORTModelForSequenceClassification.from_pretrained(check_point, export=True)
tokenizer = AutoTokenizer.from_pretrained(check_point)
ort_model.save_pretrained(onnx_output_dir)
tokenizer.save_pretrained(onnx_output_dir)

# 使用ONNX Runtime来加载ONNX格式文件，后端使用cpu运行
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
def create_model_for_provider(model_path: str, provider: str):
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session

onnx_model = create_model_for_provider(onnx_output_dir + "/model.onnx", "CPUExecutionProvider")
from datasets import load_dataset
dataset = load_dataset("clinc_oos", "plus")
intents = dataset["train"].features["intent"]

# 在onnxruntime中定义一个类似pipeline管道
from scipy.special import softmax
class OnnxPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        outputs = self.model.run(None, inputs)
        logits = outputs[0][0]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        return [{
            "label": intents.int2str(pred_idx),
            "score": probs[pred_idx]
        }]

onnx_pipe = OnnxPipeline(onnx_model, tokenizer)
print(onnx_pipe(dataset["test"]["text"][1]))
from chapter_8_3_onnx_perfmance import ONNXPerformanceBenchmark
onnx_benchmark = ONNXPerformanceBenchmark(onnx_pipe, dataset["test"], model_path=onnx_output_dir + "/model.onnx")
onnx = onnx_benchmark.run()
print(onnx)
# [{'label': 'translate', 'score': np.float32(0.9407314)}]
# Model size: 255.98 MB
# Latency mean: 13.9357 ± 2.2238 ms
# Accuracy: 0.8822
# {'opt_ONNX_Baseline': {'size': {'model_size': 255.98189163208008}, 'time': {'time_avg': np.float64(13.935740000742953), 'time_std': np.float64(2.223766362353139)}, 'accuracy': {'accuracy': 0.8821818181818182}}}

from chapter_8_1 import PerformanceBenchmark
from transformers import pipeline
pipe = pipeline("text-classification", model="transformersbook/bert-base-uncased-finetuned-clinc", device="cpu")
original_benchmark = PerformanceBenchmark(pipe, dataset["test"], optim_type="model_original")
original = original_benchmark.run()
pipe = pipeline("text-classification", model="./chapter_8/chapter_8_2/baseline", device="cpu")
distillation_benchmark = PerformanceBenchmark(pipe, dataset["test"], optim_type="model_distillation")
distillation = distillation_benchmark.run()
pipe = pipeline("text-classification", model=r"./chapter_8/chapter_8_2/opt", device="cpu")
opt_benchmark = PerformanceBenchmark(pipe, dataset["test"], optim_type="opt_distill_model")
opt = opt_benchmark.run()
import torch
from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("./chapter_8/chapter_8_2/opt")
model_quantize = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
tokenizer_quan = AutoTokenizer.from_pretrained("./chapter_8/chapter_8_2/quantize_model")
quantize_state = torch.load("./chapter_8/chapter_8_2/quantize_model/quantized_model.pt")
# 加载量化的具体配置（相当于静态量化）
model_quantize.load_state_dict(quantize_state)
pipe = pipeline("text-classification", model=model_quantize, tokenizer=tokenizer_quan, device="cpu")
quantize_benchmark = PerformanceBenchmark(pipe, dataset["test"], optim_type="quantize_opt_distill_model")
# Model size: 132.40 MB
quantize = quantize_benchmark.run()

import matplotlib.pyplot as plt

plt.ylim(80,95)
plt.xlim(opt["opt_distill_model"]["time"]["time_avg"] - 5, original["model_original"]["time"]["time_avg"] + 5)
plt.scatter(
    original["model_original"]["time"]["time_avg"],
    original["model_original"]["accuracy"]["accuracy"] * 100,
    label="model_original"
)
plt.scatter(
    distillation["model_distillation"]["time"]["time_avg"],
    distillation["model_distillation"]["accuracy"]["accuracy"] * 100,
    label="model_distillation"
)
plt.scatter(
    opt["opt_distill_model"]["time"]["time_avg"],
    opt["opt_distill_model"]["accuracy"]["accuracy"] * 100,
    label="opt_distill_model"
)
plt.scatter(
    quantize["quantize_opt_distill_model"]["time"]["time_avg"],
    quantize["quantize_opt_distill_model"]["accuracy"]["accuracy"] * 100,
    label="quantize_opt_distill_model"
)

plt.scatter(
    onnx["opt_ONNX_Baseline"]["time"]["time_avg"],
    onnx["opt_ONNX_Baseline"]["accuracy"]["accuracy"] * 100,
    label="opt_ONNX_Baseline"
)
plt.xlabel("Time (ms)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()


# 使用ONNX Runtime自带的量化对加载进去的蒸馏模型（opt）进行量化
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(model_input=onnx_output_dir + "/model.onnx", model_output=onnx_output_dir + "/quantize_model.onnx", weight_type=QuantType.QUInt8)
quantize_onnx_model = create_model_for_provider(onnx_output_dir + "/quantize_model.onnx", "CPUExecutionProvider")
pipe = OnnxPipeline(quantize_onnx_model, tokenizer)
quantize_onnx_per = ONNXPerformanceBenchmark(pipe, dataset["test"], model_path=onnx_output_dir + "/quantize_model.onnx", optim_type="opt_ONNX_Quantize")
# Model size: 64.37 MB
quantize_onnx = quantize_onnx_per.run()

plt.ylim(80,95)
plt.xlim(opt["opt_distill_model"]["time"]["time_avg"] - 5, original["model_original"]["time"]["time_avg"] + 5)
plt.scatter(
    original["model_original"]["time"]["time_avg"],
    original["model_original"]["accuracy"]["accuracy"] * 100,
    label="model_original"
)
plt.scatter(
    distillation["model_distillation"]["time"]["time_avg"],
    distillation["model_distillation"]["accuracy"]["accuracy"] * 100,
    label="model_distillation"
)
plt.scatter(
    opt["opt_distill_model"]["time"]["time_avg"],
    opt["opt_distill_model"]["accuracy"]["accuracy"] * 100,
    label="opt_distill_model"
)
plt.scatter(
    quantize["quantize_opt_distill_model"]["time"]["time_avg"],
    quantize["quantize_opt_distill_model"]["accuracy"]["accuracy"] * 100,
    label="quantize_opt_distill_model"
)

plt.scatter(
    onnx["opt_ONNX_Baseline"]["time"]["time_avg"],
    onnx["opt_ONNX_Baseline"]["accuracy"]["accuracy"] * 100,
    label="opt_ONNX_Baseline"
)

plt.scatter(
    quantize_onnx["opt_ONNX_Quantize"]["time"]["time_avg"],
    quantize_onnx["opt_ONNX_Quantize"]["accuracy"]["accuracy"] * 100,
    label="opt_ONNX_Quantize"
)

plt.xlabel("Time (ms)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()