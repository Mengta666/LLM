from pathlib import Path
from time import perf_counter
import evaluate

import numpy as np


class ONNXPerformanceBenchmark:
    def __init__(self, pipeline, dataset, model_path, optim_type="opt_ONNX_Baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type
        self.model_path = model_path

    def compute_size(self):
        model_size = Path(self.model_path).stat().st_size / 1024 ** 2
        print(f"Model size: {model_size:.2f} MB")
        return {"model_size": model_size}
    def time_pipeline(self):
        # 暖启动
        for _ in range(10):
            self.pipeline(self.dataset["text"][0])
        latencies = []
        for index in range(100):
            start_time = perf_counter()
            self.pipeline(self.dataset["text"][index])
            end_time = perf_counter()
            latencies.append(end_time - start_time)
        latencies_mean = np.mean(latencies) * 1000
        latencies_std = np.std(latencies) * 1000
        print(f"Latency mean: {latencies_mean:.4f} ± {latencies_std:.4f} ms")
        return {"time_avg": latencies_mean, "time_std": latencies_std}
    def compute_accuracy(self):
        preds, labels = [], []
        intent = self.dataset.features["intent"]
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            preds.append(intent.str2int(pred))
            labels.append(example["intent"])
        compute_acc = evaluate.load("accuracy")
        acc = compute_acc.compute(predictions=preds, references=labels)
        print(f"Accuracy: {acc['accuracy']:.4f}")
        return {"accuracy": acc["accuracy"]}

    def run(self):
        metrics = {self.optim_type: {}}
        metrics[self.optim_type]["size"] = self.compute_size()
        metrics[self.optim_type]["time"] = self.time_pipeline()
        metrics[self.optim_type]["accuracy"] = self.compute_accuracy()
        return metrics