# 加载未处理的数据集
from datasets import load_dataset, load_from_disk

# csv 按照制表符进行分割
drug_dataset = load_dataset("csv",
                            data_files={"train": r"D:\.cache\datasets\DrugsCom_raw\drugsComTrain_raw.tsv",
                                        "test": r"D:\.cache\datasets\DrugsCom_raw\drugsComTest_raw.tsv"},
                            delimiter="\t")
# 打乱数据
drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_dataset["train"][:3])
# 判断 Unnamed是否唯一，有很多种方法，也可以使用set：len(set(drug_dataset[key]["Unnamed: 0"]))
for key in drug_dataset.keys():
    assert len(drug_dataset[key]["Unnamed: 0"]) == len(set(drug_dataset[key]["Unnamed: 0"]))
# 'Unnamed: 0' 其实是患者id，重命名一下
drug_dataset = drug_dataset.rename_column("Unnamed: 0", "patient_id")
print(drug_dataset["train"].features)

# condition的大写字母全部小写，确保唯一性
# condition 包含None字段，特殊处理，使用数据集的过滤方法filter，删除None字段对应的所有数据
drug_dataset = drug_dataset.filter(lambda x : x['condition'] is not None)
def lowercase_condition(example):
    return {'condition': example['condition'].lower()}
drug_dataset = drug_dataset.map(lowercase_condition)
print("condition小写转化完成")

# 对于超长文本，应有字数统计，使用split进行粗略统计，使用字数多的样本进行训练，增强泛化能力，
# 新增一个review_length字段，仍然使用map函数
def count_words(example):
    return {'review_length': len(example['review'].split())}
drug_dataset = drug_dataset.map(count_words)
print("超长文本字数统计处理完成")
print(drug_dataset["train"].sort("review_length", reverse=False)[:3])
# 截断，使用字数多的评论，使用filter进行筛选
drug_dataset = drug_dataset.filter(lambda x : x['review_length'] > 30)
print(drug_dataset.shape)
print("超长文本截断处理完成")

# it&#039 处理这种html的特定字符，html库处理，使用map函数进行全部映射
import html
def html_decode(example):
    return {'review_clean': html.unescape(example['review'])}
# drug_dataset = drug_dataset.map(html_decode)
# 使用分批量batch来加快处理，列表推导式比for循环快得多
drug_dataset = drug_dataset.map(lambda x : {'review_clean': [html.unescape(o) for o in x["review"]]}, batched=True)
print("html字符处理完成")
print(drug_dataset["train"][:3])


# 对评论进行分词，使用tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# 使用默认截断时，截断的部分会被直接舍弃，下面增加return_overflowing_tokens参数，
# 将超出截断的文本返回（overflow_to_sample_mapping），并使用map函数进行全部映射
print(drug_dataset["train"][0])
def tokenize_split_function(example):
    result = tokenizer(example["review_clean"], truncation=True, max_length=128, return_overflowing_tokens=True)
    sample_map = result.pop("overflow_to_sample_mapping")
    for keys, values in example.items():
        if not isinstance(values, list):
            values = [values]
        # print(f"键值对：{keys},  {values}")
        result[keys] = [values[i] for i in sample_map]
    return result
result = tokenize_split_function(drug_dataset["train"][0])
# print(result) 中 'overflow_to_sample_mapping': [0, 0] 解读
# 表示分词结果包含2个子序列（input_ids 有2个列表），这两个子序列都来自原始样本的索引 0（即第一条评论）。
# 这意味着第一条评论（drug_dataset["train"][0]["review"]）被分割成了两个子序列。
# 更一般：{
#     'input_ids': [[101, ..., 102], [101, ..., 102], [101, ..., 102], [101, ..., 102]],  # 4个子序列
#     'attention_mask': [...],
#     'overflow_to_sample_mapping': [0, 0, 1, 2]  # 子序列0和1来自样本0，子序列2来自样本1，子序列3来自样本2
# }
print(result)
print( [ len(x) for x in result["input_ids"] ] )
# 使用批次和多进程进行加快分词，由于windows系统原因，须在if __name__ == "__main__"下运行多进程
# 1. 使用remove_columns参数（remove_columns=drug_dataset["train"].column_names）删除指定字段，并使用result结果重新映射，这样会删除原始数据，
# 但可以解决报错：pyarrow.lib.ArrowInvalid: Column 8 named input_ids expected length 1000 but got length 1463.
# 2. 对tokenize_split_function进行重新映射，保留原始数据，这就意味着：同一段评论被映射为多个子序列，每个子序列包含一个片段，然而这些片段除了input_ids和另外两个分词不同外，其余的都一样。如下，其中一个子序列：
# {'input_ids': [[101, 107, 1422, 1488, 1110, 9079, 1194, 1117, 2223, 1989, 1104, 1130, 19972, 11083, 119, 1284, 1245, 4264, 1165, 1119, 1310, 1142, 1314, 1989, 117, 1165, 1119, 1408, 1781, 1103, 2439, 13753, 1119, 1209, 1129, 1113, 119, 1370, 1160, 1552, 117, 1119, 1180, 6374, 1243, 1149, 1104, 1908, 117, 1108, 1304, 172, 14687, 1183, 117, 1105, 7362, 1111, 2212, 129, 2005, 1113, 170, 2797, 1313, 1121, 1278, 12020, 113, 1304, 5283, 1111, 1140, 119, 114, 146, 1270, 1117, 3995, 1113, 6356, 2106, 1105, 1131, 1163, 1106, 6166, 1122, 1149, 170, 1374, 1552, 119, 3969, 1293, 1119, 1225, 1120, 1278, 117, 1105, 1114, 2033, 1146, 1107, 1103, 2106, 119, 1109, 1314, 1160, 1552, 1138, 1151, 2463, 1714, 119, 1124, 1110, 150, 21986, 3048, 1167, 5340, 1895, 1190, 1518, 102], [101, 119, 1124, 1110, 1750, 6438, 113, 170, 1363, 1645, 114, 117, 1750, 172, 14687, 1183, 119, 1124, 1110, 11566, 1155, 1103, 1614, 1119, 1431, 119, 8007, 1117, 4658, 1110, 1618, 119, 1284, 1138, 1793, 1242, 1472, 23897, 1105, 1177, 1677, 1142, 1110, 1103, 1211, 3903, 119, 107, 102]],
# 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
# 'patient_id': [95260, 95260], 'drugName': ['Guanfacine', 'Guanfacine'], 'condition': ['adhd', 'adhd'],
# 'review': ['"My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. \r\nWe have tried many different medications and so far this is the most effective."', '"My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. \r\nWe have tried many different medications and so far this is the most effective."'],
# 'rating': [8.0, 8.0], 'date': ['April 27, 2010', 'April 27, 2010'], 'usefulCount': [192, 192], 'review_length': [141, 141]}
tokenized_dataset = drug_dataset.map(tokenize_split_function, batched=True)
print("评论分词完成")
print(tokenized_dataset)

# 现在数据已经划分完毕，使用时还需要拼凑出完整的数据集（这里的数据被分割成了多段），或者1直接舍弃
# 将数据集转换成其他格式
tokenized_dataset.set_format("torch")
print(tokenized_dataset["train"][0])
tokenized_dataset.reset_format()
print(tokenized_dataset["train"][0])

# 将原始清理后的数据（没有分词化）划分后保存到本地磁盘
# 创建验证集，使用train_test_split方法
drug_dataset["validation"] = drug_dataset.pop("test")
print(drug_dataset)
drug_dataset_new= drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
drug_dataset_new["validation"] = drug_dataset["validation"]
print(drug_dataset_new)
print("原始数据集划分完毕")
# 保存为默认Arrow文件，使用 load_from_disk 读取
drug_dataset_new.save_to_disk("./my_datasets/drug_dataset_new_origin")
# 保存为json/csv文件
# for split, dataset in drug_dataset_new.items():
#     dataset.to_json(f"./my_datasets/drug_dataset_new_{split}.json")
#     dataset.to_csv(f"./my_datasets/drug_dataset_new_{split}.csv")
print("原始数据集保存完毕")



