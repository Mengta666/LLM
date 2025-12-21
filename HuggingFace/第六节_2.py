from transformers import AutoTokenizer

example = "My name is Sylvain and I work at Hugging Face in Brooklyn"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
encoding = tokenizer(example)

print(type(encoding))
print(encoding.tokens())
print(encoding.word_ids())

start, end = encoding.word_to_chars(3)
print(example[start:end])

# <class 'transformers.tokenization_utils_base.BatchEncoding'>
# ['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', 'and', 'I', 'work', 'at', 'Hu', '##gging', 'Face', 'in', 'Brooklyn', '[SEP]']
# [None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, None]
# Sylvain