import requests  # 用于发送HTTP请求，例如调用天气API
import configparser  # 用于读取配置文件（如API密钥）
import time  # 用于时间相关操作，例如性能监控
import sqlite3  # 用于SQLite数据库操作，存储长期记忆
import numpy as np  # 用于数值计算，例如余弦相似度
import pandas as pd  # 用于数据处理和CSV文件读取，例如天气ID列表
import asyncio  # 用于异步编程，支持并发处理查询
from functools import wraps  # 用于装饰器的高阶函数包装
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Hugging Face Transformers库，用于加载GPT-2模型和分词器
from sentence_transformers import SentenceTransformer  # 用于句子嵌入模型，实现向量搜索

# 记忆管理器类：负责长期记忆的存储和检索，使用SQLite数据库
class MemoryManager:
    @staticmethod
    def load_long_term_memory():
        """加载或创建长期记忆数据库连接和游标。
        返回：数据库连接和游标对象。
        """
        # 连接到本地SQLite数据库文件，如果不存在则创建
        conn = sqlite3.connect(r".\dataset\long_term_memory.db")
        cursor = conn.cursor()
        # 创建客户历史表，如果不存在
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_history (
        customer_id TEXT, query TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        conn.commit()  # 提交表创建操作
        return conn, cursor

    def save_to_memory(self, customer_id: str, query: str):
        """将客户查询保存到长期记忆数据库。
        参数：
            customer_id: 客户ID（字符串）
            query: 用户查询内容（字符串）
        """
        conn, cursor = self.load_long_term_memory()  # 获取数据库连接
        # 插入新记录
        cursor.execute("""
                       INSERT INTO customer_history (customer_id, query)
                       SELECT ?, ? WHERE NOT EXISTS (
                SELECT 1 FROM customer_history
                WHERE customer_id = ? AND query = ?)
        """, (customer_id, query, customer_id, query))
        conn.commit()  # 提交更改
        conn.close()  # 关闭连接

    def get_last_query(self, customer_id: str) -> str:
        """获取指定客户最近一次查询。
        参数：
            customer_id: 客户ID（字符串）
        返回：最近查询字符串，或None如果无记录。
        """
        conn, cursor = self.load_long_term_memory()  # 获取数据库连接
        # 查询最近记录，按时间降序排序，取第一条
        cursor.execute("""
        SELECT query FROM customer_history WHERE customer_id = ? ORDER BY timestamp DESC LIMIT 1
        """, (customer_id,))
        result = cursor.fetchone()  # 获取单条结果
        conn.close()  # 关闭连接
        return result[0] if result else None  # 返回查询内容或None

# 创建记忆管理器实例
memory_manager = MemoryManager()

# 性能监控装饰器：用于测量函数执行时间
def performance_monitor(func):
    """装饰器，用于监控函数执行时间并打印。
    参数：
        func: 被装饰的函数
    返回：包装后的函数。
    """
    @wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.time()  # 记录结束时间
        print(f"{func.__name__} 运行花费 {end_time - start_time:.2f} s")  # 打印执行时间
        return result
    return wrapper

# 模型管理器类：负责加载和使用GPT-2生成文本
class ModelManager:
    def __init__(self):
        """初始化模型管理器，加载分词器和模型。"""
        self.tokenizer, self.model = self.load_model_and_tokenizer()  # 加载模型和分词器

    @staticmethod
    def load_model_and_tokenizer():
        """静态方法：加载GPT-2分词器和模型。
        返回：分词器和模型对象。
        """
        print("Loading model and tokenizer...")  # 打印加载提示
        # 从Hugging Face加载GPT-2分词器
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # 加载GPT-2语言模型，并设置填充token ID以避免警告
        model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
        print("Model and tokenizer loaded.")  # 打印加载完成
        return tokenizer, model

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """使用GPT-2模型生成文本回复。
        参数：
            prompt: 输入提示词（字符串）
            max_length: 生成文本最大长度（默认50）
        返回：生成的文本字符串。
        """
        # 将提示词编码为PyTorch张量
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        # 生成输出序列
        output = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        # 解码输出，去除特殊token
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# 创建模型管理器实例
model_manager = ModelManager()

# 向量搜索类：基于句子嵌入实现知识库搜索
class VectorSearch:
    def __init__(self):
        """初始化向量搜索，使用SentenceTransformer模型和知识库。"""
        # 加载句子嵌入模型
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        # 定义知识库（可扩展为从向量数据库加载）
        self.knowledge_base = [
            "股票市场是风险投资的主要渠道。",
            "债券投资适合保守型投资者。",
            "外汇市场波动大，适合有经验的投资者参与。"
        ]
        # 计算知识库向量的嵌入
        self.knowledge_vectors = self.model.encode(self.knowledge_base)

    @staticmethod
    def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """计算两个向量的余弦相似度。
        参数：
            vector_a, vector_b: 输入向量（NumPy数组）
        返回：相似度分数（浮点数，范围[-1, 1]）。
        """
        return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

    def search(self, query: str) -> str:
        """在知识库中搜索最相似的条目。
        参数：
            query: 查询字符串
        返回：最匹配的知识库条目。
        """
        # 计算查询向量的嵌入
        query_vector = self.model.encode([query])[0]
        # 计算与所有知识向量的相似度
        similarity_list = [self.cosine_similarity(query_vector, vector) for vector in self.knowledge_vectors]
        # 找到最大相似度的索引
        index = np.argmax(similarity_list)
        return self.knowledge_base[index]

# 创建向量搜索实例
vector_search = VectorSearch()

# 读取配置文件
config = configparser.ConfigParser()
# 读取API配置文件，指定UTF-8编码
config.read(r".\config\all_apis.ini", encoding="utf-8")
# 获取百度天气API密钥
YOUR_AK = config["百度天气AK"]["YOUR_AK"]

# API管理器类：负责外部API调用，如天气查询
class APIManager:
    @staticmethod
    def get_weather(city_id: str) -> str:
        """获取指定城市ID的天气信息。
        参数：
            city_id: 城市区划ID（字符串，例如'110100'为北京）
        返回：天气描述字符串，或错误消息。
        """
        # 构建百度天气API URL
        url = f"https://api.map.baidu.com/weather/v1/?district_id={city_id}&data_type=all&ak={YOUR_AK}"
        # 发送GET请求
        response = requests.get(url)
        # 解析JSON响应
        data = response.json()
        # 检查API状态，如果非0则返回错误
        if data["status"] != 0:
            return "无法获取天气信息"
        # 提取位置、天气和温度信息
        location = data["result"]["location"]["country"] + data["result"]["location"]["city"]
        weather = data["result"]["now"]["text"]
        temperature = data["result"]["now"]["temp"]
        # 返回格式化天气字符串
        return f"{location}的实时天气是{weather}，实时温度是 {temperature} °C"

# print(APIManager.get_weather("110100"))  # 示例调用（注释掉以避免直接执行）

# 金融智能Agent类：核心处理用户查询
class FinancialAgent:
    @performance_monitor  # 应用性能监控装饰器
    async def handle_query(self, customer_id: str, query: str) -> str:
        """异步处理用户查询，支持天气、投资建议等功能。
        参数：
            customer_id: 客户ID（字符串）
            query: 用户查询（字符串）
        返回：处理结果字符串（实际通过print输出）。
        """
        # 保存查询到记忆
        memory_manager.save_to_memory(customer_id, query)
        # 检查查询类型
        if "天气" in query:
            # 硬编码城市为北京，后续可集成NER提取
            city = "遵义"
            try:
                # 读取天气ID CSV文件
                df = pd.read_csv(r".\一些参照数据\百度天气ID列表\weather_district_id.csv")
                # 查询城市对应的district_id
                result = df.query(f"district == '{city}'")["district_id"].iloc[0]
            except Exception:  # 捕获任何异常
                print("无法获取到天气信息")
            else:
                # 调用天气API并打印结果
                print(APIManager.get_weather(result))
        elif "投资建议" in query:
            # 使用向量搜索获取建议
            advice = vector_search.search(query)
            print(f"智能投资建议：{advice}")
        else:
            # 使用GPT-2生成通用回复
            response = model_manager.generate_text(query)
            print(f"智能助手回复：{response}")

# 主函数：异步运行测试查询
async def main():
    """主入口：创建Agent并测试多个查询。"""
    agent = FinancialAgent()  # 创建Agent实例
    # 测试天气查询
    await agent.handle_query("customer_001", "今天北京天气如何？")
    # 测试股票投资查询
    await agent.handle_query("customer_001", "股票市场如何投资？")
    # 测试外汇投资查询（英文）
    await agent.handle_query("customer_001", "Advisor what is the advice for investment in foreign exchange?")
    # 测试通用投资建议查询
    await agent.handle_query("customer_001", "给我一些投资建议。")

# 程序入口：运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())