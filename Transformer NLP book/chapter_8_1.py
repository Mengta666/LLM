import os
import time
from datasets import load_dataset
from transformers import pipeline
from pathlib import Path
import evaluate
import torch

class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="model_baseline"):
        """
            初始化性能基准测试类。

            参数:
                pipeline: transformers 的 pipeline 对象，用于文本分类。
                dataset: 测试数据集，包含文本和意图标签。
                optim_type: 优化类型，默认为 "model_baseline"，用于区分不同模型配置。
        """
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        """
            计算模型在测试数据集上的准确率。

            返回:
                dict: 包含准确率的字典，例如 {'accuracy': 0.95}。
        """
        preds, labels = [], []
        intent = self.dataset.features["intent"]
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            preds.append(intent.str2int(pred))
            labels.append(example["intent"])
        compute_accuracy = evaluate.load("accuracy")
        acc = compute_accuracy.compute(predictions=preds, references=labels)
        # print(f"Accuracy: {acc}")
        return acc

    def compute_size(self):
        """
            计算模型大小，通过保存模型状态字典到磁盘并测量文件大小。

            返回:
                float: 模型大小（单位：MB）。
        """
        # 通过torch的save()函数将模型序列化到磁盘来计算模型大小
        # save()函数保存的是model的状态字典，将模型的每个层映射到其可以学习的参数
        state_dict = self.pipeline.model.state_dict()
        torch.save(state_dict, Path("model_state_dict.pt"))
        # Path方法中的stat()方法返回文件或目录的元数据，包括文件大小，创建时间，修改时间等
        size = Path("model_state_dict.pt").stat().st_size / 1024 ** 2
        print(f"Model size: {size:.2f} MB")
        time.sleep(3)
        os.remove("model_state_dict.pt")
        return size

    def time_pipeline(self):
        """
            计算模型的平均推理延迟（每个样本的处理时间）。

            返回:
                dict: 包含平均延迟（time_avg）和标准差（time_std），单位为毫秒。
        """
        # 计算延迟：即每个输入模型处理后并返回预测结果的平均时间
        latencies = []
        # 暖启动一下
        for _ in range(10):
            self.pipeline(self.dataset[0]["text"])
        # 计算延迟
        for i in range(100):
            start_time = time.perf_counter()
            _ = self.pipeline(self.dataset[i]["text"])
            latencies.append(time.perf_counter() - start_time)
        import numpy as np
        time_avg = np.mean(latencies) * 1000
        time_std = np.std(latencies) * 1000
        # print(f"Average time: {time_avg:.2f} ± {time_std:.2f} ms")
        return {"time_avg": time_avg, "time_std": time_std}

    def run(self):
        """
            运行所有性能测试，收集模型大小、推理时间和准确率。

            返回:
                dict: 包含所有性能指标的嵌套字典。
        """
        metrics = {self.optim_type: {}}
        metrics[self.optim_type]["size"] = self.compute_size()
        metrics[self.optim_type]["time"] = self.time_pipeline()
        metrics[self.optim_type]["accuracy"] = self.compute_accuracy()
        return metrics

if __name__ == "__main__":
    dataset = load_dataset("clinc_oos", "plus")
    print(dataset["train"].features)
    print(dataset["train"]["intent"][0])
    print(dataset["train"].features["intent"].int2str(61))
    pipe = pipeline("text-classification", model="transformersbook/bert-base-uncased-finetuned-clinc")
    query = "Hey, I'd like to ren a vehicle from Nov 1st to Nov 15th in Paris and I ned a 15 passenger van"
    print(pipe(query))
    compute = PerformanceBenchmark(pipe, dataset["test"])
    # compute.compute_accuracy()
    # compute.compute_size()
    # compute.time_pipeline()
    print(compute.run())
