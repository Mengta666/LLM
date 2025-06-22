from transformers import pipeline
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import AutoModelForSequenceClassification
from torch import tensor


checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
inputs = ["Using a Transformer network is simple",
          "when you use the pretrained model from HuggingFace"]
outputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
print(outputs)
for i, input in enumerate(outputs["input_ids"]):
    print(f"原始输入: {tokenizer.decode(input)}")
input_text = tokenizer.batch_decode(outputs["input_ids"])
print(input_text)
pre_outputs = model(**outputs)
# print(pre_outputs.last_hidden_state.shape)
print(pre_outputs.logits)