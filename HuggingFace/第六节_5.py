import tokenizers.models as models
from tokenizers import decoders, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import os
from datasets import load_dataset

# 设置数据集
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

def get_train_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i+1000]["text"]

# 1. wordpiece
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 标准化,NFD与Strip Accents 去掉重音符,Lowercase 小写化
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

# 测试标准化: hello how are u?
print(tokenizer.normalizer.normalize_str("Hello how are ü?"))

# 预分词,Whitespace 使用
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# 当然,也可以按照标准化一样设置分词格式
# tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])

# 测试预分词效果:
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

# 训练
# 训练前,需要设置分词中的特殊字符有哪些(例如 CLS], [SEP], [PAD], [UNK], [MASK]),这些在词汇表中不会又分词得到,必须手动指定
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# 使用迭代器进行训练词汇表
tokenizer.train_from_iterator(get_train_corpus(), trainer=trainer)

# 使用 encode 进行分词测试
print(tokenizer.encode("Let's test my pre-tokenizer.").tokens)

# 对于原始的 tokenizer(利用AutoTokenizer 定义的),有[CLS],[SEP]标识,需要定义添加,也就是后处理(processors)
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

# 定义一个后处理模板
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
)

encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens, encoding.type_ids)

# 定义一个解码器,也就是将编码后的分词id进行解码为正常的句子
tokenizer.decoder = decoders.WordPiece(prefix="##")
print(tokenizer.decode(encoding.ids))

# ok,一个自定义分词器训练完成,保存到本地
if not os.path.exists("tokenizer"):
    os.mkdir("tokenizer")
tokenizer.save(r"tokenizer\wordpiece.json")

# 加载这个分词器
new_tokenizer = Tokenizer.from_file(r"tokenizer\wordpiece.json")

# 将训练后的分词器载入到一个通用类中,以便能在 transformers 中使用(也就是能以 AutoTokenizer 定义的方式使用):
from transformers import PreTrainedTokenizerFast
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", #也可以从 tokenizer 文件中加载
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# 2. BPE,使用字节级进行训练,这将不存在未知字符
tokenizer = Tokenizer(models.BPE())
# 字节级不需要标准化

# 预分词,add_prefix_space=False 不在开头添加空格,默认为True
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!"))

# 训练
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_train_corpus(), trainer=trainer)

# 测试分词
print(tokenizer.encode("Let's test this tokenizer.").tokens)

# 后处理, 将Ġ作为单词的一部分, 而不忽略Ġ(没偏移), 也就是说, 截取某一个分词时, 若分词前有空格,那么这个空格也必须取到
# 字节级 BPE 不强制需要设置 <CLS> 和 <SEP>，因为这些标记是针对特定 NLP 任务（如 BERT）的，而不是分词算法本身的要求。
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
print(sentence[start:end]) # _test (_表示空格)

# 解码器
tokenizer.decoder = decoders.ByteLevel()
print(tokenizer.decode(encoding.ids)) # Let's test this tokenizer.

tokenizer.save(r"tokenizer\BPE.json")

# 同样, 若要在 transformers 中使用这个 tokenizer
# from transformers import PreTrainedTokenizerFast
# BPE_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     bos_token="<|endoftext|>",
#     eos_token="<|endoftext|>",
# )

# 3. unigram, 在训练中需设置未知 token
# Unigram 模型的目标是找到一个最优的子词集合，使得分词后的文本表示最大化数据的似然（likelihood）。
tokenizer = Tokenizer(models.Unigram())
# 标准化
from tokenizers import Regex
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " ") # " {2,}" 表示匹配两个或多个（连续的）空格。替换为单个空格
    ]
)

# 预分词, Metaspace 预分词器会将输入文本按空格（或特殊的 metaspace 标记，通常是 _）分割，同时保留空格信息。
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test this tokenizer."))

# 训练
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000,
    special_tokens=special_tokens, unk_token="<unk>"
)
tokenizer.train_from_iterator(get_train_corpus(), trainer= trainer)
encoding = tokenizer.encode("Let's test the tokenizer")
print(encoding.tokens)
print(encoding.type_ids)
cls_token_id = tokenizer.token_to_id("<cls>")
sep_token_id = tokenizer.token_to_id("<sep>")
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"$A:0 <sep>:0 <cls>:2",
    pair=f"$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<cls>", cls_token_id), ("<sep>", sep_token_id)]
)

encoding = tokenizer.encode("Let's test the tokenizer...", "on a pair of sentences!")
print(encoding.tokens)
print(encoding.type_ids)
tokenizer.decoder = decoders.Metaspace()
tokenizer.save(r"tokenizer\Unigram.json")