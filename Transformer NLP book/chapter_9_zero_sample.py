from transformers import pipeline

pipe = pipeline("fill-mask", model="bert-base-uncased")
text = "The main characters of the movie madacascar are a lion, a zebra, a giraffe, and a hippo. "
prompt = "the movie is about [MASK]."
outputs = pipe(text+prompt)
# Token animals:	0.103%
# Token lions:	0.066%
# Token birds:	0.025%
# Token love:	0.015%
# Token hunting:	0.013%
for i in outputs:
    print(f"Token {i['token_str']}:\t{i['score']:.3f}%")

from transformers import AutoModel, AutoTokenizer
checkpoint = "miguelvictor/python-gpt2-large"
model = AutoModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer("I took my dog for a walk", padding= "max_length", max_length=64, return_tensors="pt")
outputs = model(**inputs)
print(inputs)
print(outputs["last_hidden_state"].shape)
import torch
def mean_poolng(outputs, attention_mask):
    token_embeddings = outputs[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # torch.Size([1, 64, 1280])
    # print(input_mask_expanded.shape)
    # 逐元素相乘
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # torch.Size([1, 1280])
    # print(sum_embeddings.shape)
    return sum_embeddings / sum_mask

mean_poolng(outputs, inputs["attention_mask"])



