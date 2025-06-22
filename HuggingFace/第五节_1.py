from datasets import load_dataset


# 这个是已处理的数据集，加载后直接用tokenizer处理即可
squad_it_dataset = load_dataset("json",
                                data_files={"train": r"D:\.cache\datasets\squad-it\SQuAD_it-train.json",
                                            "test": r"D:\.cache\datasets\squad-it\SQuAD_it-test.json"},
                                field = "data")
print(squad_it_dataset["train"][0])