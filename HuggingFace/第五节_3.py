import logging
import json
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import requests
from datasets import load_dataset
from multiprocessing import Value, Lock


GITHUB_TOKEN = "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"  #  将你的GitHub令牌复制到此处
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

# 爬取github问题页，直接复制的官网
def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)
    batch = []
    all_issues = []
    per_page = 100  ## 每页返回的 issue 的数量
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"
    for page in tqdm(range(num_pages)):
        # 使用 state=all 进行查询来获取 open 和 closed 的issue
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())
        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # 重置batch
            print(f"Reached GitHub rate limit. Sleeping for one minute ...")
            time.sleep(60)
    all_issues.extend(batch)
    # 从已有数据创建一个 DataFrame，这个DataFrame 可以直接保存为 JSONL 文件
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )

# 读取下载的数据，数据中有一个"closed_at": null, datasets推断其为时间，需要特殊处理，只要部分字段
def clean_issues(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line.strip())
            keys = ["number", "title", "comments_url", "url", "repository_url", "labels_url", "events_url", "html_url", "id", "user", "body"]
            # 只需要上述字段的数据
            new_dict = {key: record[key] for key in keys}
            f_out.write(json.dumps(new_dict, ensure_ascii=False) + "\n")

# 提取评论数据
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=r".\github\huggingface\script.log",
    encoding="utf-8",
    filemode="a"
)
logger = logging.getLogger(__name__)
# 共享计数器和锁
request_counter = Value("i", 0)  # 多进程共享的请求计数器
counter_lock = Lock()  # 确保计数器更新线程安全
REQUEST_LIMIT = 4000  # 每 1000 次请求暂停
PAUSE_DURATION = 3600  # 暂停 1 小时（3600 秒）

def extract_comments(example):
    """提取 issue 的评论数据，带请求计数和暂停逻辑"""
    global request_counter, counter_lock
    # 检查是否需要暂停
    with counter_lock:
        if request_counter.value >= REQUEST_LIMIT:
            logger.info(f"已到请求限制： {REQUEST_LIMIT} 等待 {PAUSE_DURATION} seconds...")
            time.sleep(PAUSE_DURATION)
            request_counter.value = 0  # 重置计数器
            logger.info("等待结束，再次获取")

    # 发送 API 请求
    try:
        with counter_lock:
            request_counter.value += 1
            current_count = request_counter.value
        logger.debug(f"开始第{current_count}次请求， 抓取评论： {example['comments_url']}")
        res = requests.get(example["comments_url"], headers=headers)
        if res.status_code != 200:  # 速率限制
            logger.warning("出现速率限制，等待 3600 seconds...")
            with counter_lock:
                request_counter.value = 0
            time.sleep(3600)
            res = requests.get(example["comments_url"], headers=headers)
        res.raise_for_status()
        body = [item.get("body") for item in res.json()]
        return {"comments": body}
    except Exception as e:
        logger.error(f"Error fetching {example['comments_url']}: {e}")
        return {"comments": []}

if __name__ == "__main__":
    # fetch_issues()
    # 输入输出文件路径
    input_file = r".\github\huggingface\datasets-issues.jsonl"
    output_file = r".\github\huggingface\datasets-issues-new.jsonl"
    # clean_issues(input_file, output_file)
    #
    # 加载数据集
    logger.info("加载数据...")
    issues_dataset = load_dataset("json", data_files=output_file, split="train")
    logger.info(f"已加载 {len(issues_dataset)} 个数据.")
    print(issues_dataset[0])

    # 提取评论数据
    logger.info("利用4个进程获取评论数据...")
    issues_dataset = issues_dataset.map(extract_comments, num_proc=4)
    logger.info("获取评论完成.")
    print(issues_dataset[0])

    # 保存处理后的数据集
    issues_dataset.save_to_disk(r".\github\huggingface\datasets-issues-with-comments")
    issues_dataset.to_json(r".\github\huggingface\datasets-issues-with-comments.jsonl")
    logger.info("已将数据集保存到磁盘.")

    # 读取新的数据，并删除评论为空的数据，保存清理的数据，并上传到 huggingface
    issues_dataset = load_dataset("json", data_files=r".\github\huggingface\datasets-issues-with-comments.jsonl", split="train")
    issues_dataset = issues_dataset.filter(lambda x: len(x["comments"]) > 0)
    issues_dataset.to_json(r".\github\huggingface\datasets-issues-with-comments-clean.jsonl")
    # from huggingface_hub import HfApi
    # import os
    # api = HfApi(token=os.getenv("HF_TOKEN"))
    # api.upload_file(
    #     path_or_fileobj=r".\github\huggingface\datasets-issues-with-comments-clean.jsonl",  # 本地文件路径
    #     path_in_repo="datasets-issues-with-comments-clean.jsonl",  # 文件在仓库中的目标路径
    #     repo_id="mengta666/huggingface_github",  # 仓库 ID
    #     repo_type="dataset",  # 仓库类型
    # )
    #
    print(issues_dataset)