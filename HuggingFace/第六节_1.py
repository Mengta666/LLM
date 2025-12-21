from datasets import load_dataset

# 示例:使用机器语言来建立一个自定义分词器,这里使用python 语料库
file_dir = r"D:\.cache\datasets\code_search_net\python\final\jsonl"

# 指定 train test、valid文件路径
data_files = {
    "train": [f"{file_dir}/train/python_train_{i}.jsonl.gz" for i in range(14)], # n是 train 文件数量
    "test": f"{file_dir}/test/python_test_0.jsonl.gz",
    "valid": f"{file_dir}/valid/python_valid_0.jsonl.gz"
}

raw_datasets = load_dataset("json", data_files=data_files)
print(raw_datasets["train"])
print(raw_datasets["train"][123456]["original_string"])

# 利用生成器的特性,对original string 对应的文本组合成列表,训练分词器时分批训练
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        if len(dataset) - start_idx <= 1000:
            end_idx = len(dataset)
        else:
            end_idx = start_idx + 1000
        yield dataset[start_idx:end_idx]["original_string"]

training_corpus = get_training_corpus()

# 训练一个新的分词器,以一个比较全面用gpt2 预训练模型
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
example = """def add_numbers(a, b):
    '''Add the two numbers a and b'''
    return a + b"""

# 使用没有微调的分词器进行分类,参照组,输出中分词器的初始参数(进行微调),这里选
# Ċ 表示换行, Ġ 表示空格
print(old_tokenizer.tokenize(example))

# 微调分词器, 52000表示新训练的分词器的词汇量大小, (也就是这个分词器里面最多有多少个词汇)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

print(tokenizer.tokenize(example))
# 可见新的分词器分出来的效果更好

# 保存这个分词器
tokenizer.save_pretrained("code-search-net-tokenizer")
# 发送到仓库
# tokenizer.push_to_hub("code-search-net-tokenizer")