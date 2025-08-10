from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 用 GPT2Model: 提取 GPT-2 的中间表示（hidden states）用于其他任务。自定义输出层（如用于分类或回归）。
# 用 GPT2LMHeadModel：直接进行文本生成或语言建模。开箱即用的生成能力（如 model.generate()）
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

input = "I enjoy walking in the rain, "
input_ids = tokenizer.encode(input, return_tensors='pt')
# 贪婪搜索
output = model.generate(input_ids, max_length=40)
print("Output:\n"+ "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# Output:
# ----------------------------------------------------------------------------------------------------
# I enjoy walking in the rain, but I'm not sure if I'm going to be able to do it. I'm not sure if I'm going to be able to do it.

# 波束搜索
output = model.generate(input_ids, max_length=40, num_beams=5, early_stopping=True)
print("-"*100 + "\n" + "Output:\n" + "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# Output:
# ----------------------------------------------------------------------------------------------------
# I enjoy walking in the rain, but I don't like to walk in the rain. I like to walk in the rain, but I don't like to walk in the rain
# 具有词元频率限制的波束搜索
output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=False, no_repeat_ngram_size=2)
print("-"*100 + "\n" + "Output:\n" + "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# Output:
# ----------------------------------------------------------------------------------------------------
# I enjoy walking in the rain, but I don't think I'll ever be able to get out of it. I'm not sure if it's because of the weather, or if I just want to go out and do something else.

# 简单采样, 不使用top_k采样
output = model.generate(input_ids, max_length=40, do_sample=True, top_k=0)
print("-"*100 + "\n" + "Output:\n" + "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# I enjoy walking in the rain, but I also devoured your Adventure Island food T-shirts I got while wandering aroundberries and cold weather horrible [9/26/15]: Didn't

# 加入温度的简单采样
output = model.generate(input_ids, max_length=50, do_sample=True, top_k=0, temperature=0.7)
print("-"*100 + "\n" + "Output:\n" + "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# I enjoy walking in the rain, but I think as long as you don't get caught, you'll get nice and clean (and made of poop). I have been to both airports, but this one is all about SMBC. I

# top_k采样
output = model.generate(input_ids, max_length=50, do_sample=True, top_k=50)
print("-"*100 + "\n" + "Output:\n" + "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# I enjoy walking in the rain, but that's also about what you use. I'd like people to remember how important a gift it was to me when I started to care about getting through it all.

# top_p采样
output = model.generate(input_ids, max_length=50, do_sample=True, top_p=0.92)
print("-"*100 + "\n" + "Output:\n" + "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# I enjoy walking in the rain, it reminds me that we live in the land of the free, and our freedom is free to walk in the rain . And so I'm going to go look at the

# top_k和top_p结合
output = model.generate(input_ids, max_length=50, do_sample=True, top_p=0.95, top_k=50)
print("-"*100 + "\n" + "Output:\n" + "-"*100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# I enjoy walking in the rain, a sign of being at peace with myself. Being able to experience the pain of losing the one you love and where you are at, without having to think about why you might not be able to find other



