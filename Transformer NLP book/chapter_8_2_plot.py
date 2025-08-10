from chapter_8_1 import PerformanceBenchmark

from transformers import pipeline

def plot_results(dataset, path="./chapter_8/chapter_8_2/baseline"):
    dataset = dataset
    tea_pipe = pipeline("text-classification", model="transformersbook/bert-base-uncased-finetuned-clinc")
    stu_pipe = pipeline("text-classification", model=path)

    stu_benchmark = PerformanceBenchmark(pipeline=stu_pipe, dataset=dataset["test"], optim_type="distillation")
    tea_benchmark = PerformanceBenchmark(pipeline=tea_pipe, dataset=dataset["test"])

    stu_acc = stu_benchmark.run()
    print(f"Student: {stu_acc}")
    tea_acc = tea_benchmark.run()
    print(f"Teacher: {tea_acc}")

    from matplotlib import pyplot as plt

    plt.ylim(80, 90)
    plt.xlim(stu_acc["distillation"]["time"]["time_avg"] - 3, tea_acc["model_baseline"]["time"]["time_avg"] + 3)
    plt.scatter(
        stu_acc["distillation"]["time"]["time_avg"],
        stu_acc["distillation"]["accuracy"]["accuracy"] * 100,
        label="Distillation"
    )
    plt.scatter(
        tea_acc["model_baseline"]["time"]["time_avg"],
        tea_acc["model_baseline"]["accuracy"]["accuracy"] * 100,
        label="Model Baseline"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.legend()  # 添加图例
    plt.show()