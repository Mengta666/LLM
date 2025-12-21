import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from get_api_config import GetApiConfig

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# 抑制 Google API 警告
os.environ["GRPC_VERBOSITY"] = "ERROR"
logging.getLogger("google").setLevel(logging.ERROR)

# 自定义回调用于调试
class CustomCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"[Chain Start] Inputs = {inputs}")
    def on_chain_end(self, outputs, **kwargs):
        print(f"[Chain End] Outputs = {outputs}")
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LLM Start] Prompts = {prompts}")
    def on_llm_end(self, response, **kwargs):
        print(f"[LLM End] Response = {response}")

# 初始化
config = GetApiConfig()
llm = ChatGoogleGenerativeAI(
    api_key=config.get_gemini_api_key(1),
    model="models/gemini-2.0-flash",
    temperature=0.7,
    timeout=30,
    max_retries=2
)

# 定义 chain
keyword_prompt = ChatPromptTemplate.from_template(
    """
    从下面的句子提取关键词：{question}
    输出格式：以逗号分隔的关键词列表，例如 "关键词1, 关键词2"
    """
)
keyword_chain = keyword_prompt | llm | StrOutputParser()

result_prompt = ChatPromptTemplate.from_template(
    """
    基于以下关键词：{keywords}，生成一个详细的解决方案。
    """
)
result_chain = result_prompt | llm | StrOutputParser()

# 创建 pipeline
# {"keywords": lambda x: x} 是一个映射操作，将前一链的输出映射到 search_chain 期望的输入字典 {keywords: ...}。
# 可以使用 RunnablePassthrough：替换 lambda x: x 以提高代码可读性
pipeline = RunnableSequence(keyword_chain | {"keywords": RunnablePassthrough()} | result_chain)

# 执行并捕获错误
try:
    result = pipeline.invoke(
        {"question": "请给我一个关于如何使用langchain的解决方案"},
        config={"callbacks": [CustomCallback()]}  # 通过 config 添加回调
    )
    print(f"Final Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# ## 基于 Langchain 的解决方案：构建智能应用的蓝图
# Langchain 是一个强大的框架，旨在简化使用大型语言模型 (LLMs) 构建应用程序的过程。它提供了一套工具、组件和接口，可以轻松地将 LLMs 集成到各种任务中，例如问答、文档摘要、聊天机器人、代码生成等等。
#
# 以下是一个详细的解决方案，阐述如何利用 Langchain 构建智能应用，并涵盖关键概念、组件和最佳实践：
#
# **1. 理解 Langchain 的核心概念:**
#
# *   **Models (模型):** Langchain 提供了与各种 LLMs (如 OpenAI 的 GPT 系列、Google 的 PaLM、开源模型如 Llama 2) 的接口。 它允许你轻松地切换和比较不同模型，并根据你的特定需求选择最佳模型。
# *   **Prompts (提示):**  Langchain 提供了用于构建和管理提示的工具。提示是传递给 LLM 的输入，用于指导模型的行为和生成期望的输出。有效的提示工程是获得高质量结果的关键。
# *   **Chains (链):**  链是连接多个组件（如模型、提示、文档存储）以创建更复杂的应用程序的序列。 链允许你将 LLM 与其他数据源和工具组合在一起，实现更高级的功能。
# *   **Indexes (索引):** Langchain 提供了创建和查询文档索引的工具，用于从大量文本数据中检索相关信息。这对于构建问答系统和知识库至关重要。
# *   **Agents (代理):**  代理是使用 LLM 来决定采取哪些行动的系统。  它们可以访问各种工具（如搜索引擎、数据库、API），并根据 LLM 的推理选择最合适的工具来完成任务。
# *   **Memory (记忆):**  Langchain 提供了用于管理 LLM 应用程序中的对话历史记录和上下文的工具。这对于构建具有持久性的聊天机器人至关重要。
#
# **2. 确定你的应用场景和目标:**
#
# 在开始构建之前，明确你想要解决的问题或实现的目标至关重要。 例如：
#
# *   **问答系统:**  构建一个可以回答有关特定文档或知识库的问题的系统。
# *   **文档摘要:**  自动生成长篇文档的简洁摘要。
# *   **聊天机器人:**  创建一个可以进行自然语言对话的机器人。
# *   **代码生成:**  根据自然语言描述生成代码片段。
# *   **数据分析:**  使用 LLM 来理解和分析数据，并生成见解。
# *   **自动化工作流程:**  将 LLM 集成到工作流程中，以自动化任务和提高效率。
#
# **3.  选择合适的 Langchain 组件:**
#
# 根据你的应用场景，选择最合适的 Langchain 组件：
#
# *   **Model (模型):**
#     *   **OpenAI (GPT-3.5, GPT-4):**  适用于广泛的任务，提供强大的性能，但需要付费 API 密钥。
#     *   **Google PaLM:**  提供类似的能力，可能在某些特定领域表现更好。
#     *   **Hugging Face Hub:**  提供大量开源模型，可以免费使用或微调。
#     *   **选择标准:** 考虑模型的性能、成本、速度和特定领域知识。
#
# *   **Prompt (提示):**
#     *   **PromptTemplate:**  用于创建可重用的提示模板，可以轻松地插入变量。
#     *   **Few-shot prompting:**  通过在提示中提供几个示例来指导模型的行为。
#     *   **Chain-of-thought prompting:**  鼓励模型逐步推理，提高复杂问题的准确性。
#     *   **选择标准:**  根据任务的复杂性和所需的控制程度选择合适的提示策略。
#
# *   **Chain (链):**
#     *   **LLMChain:**  连接 LLM 和提示，用于生成文本。
#     *   **SequentialChain:**  按顺序执行多个链。
#     *   **RetrievalQAChain:**  从文档中检索相关信息并使用 LLM 回答问题。
#     *   **选择标准:**  根据应用所需的逻辑流程选择合适的链类型。
#
# *   **Index (索引):**
#     *   **Document Loaders:**  用于从各种来源（如文件、网页、数据库）加载文档。
#     *   **Text Splitters:**  将长文本分割成更小的块，以便进行索引。
#     *   **Vector Stores:**  将文本块转换为向量嵌入，以便进行语义搜索。  例如： Chroma, FAISS, Pinecone.
#     *   **Retrievers:**  从向量存储中检索与查询相关的文档。
#     *   **选择标准:**  根据数据源、文本大小和搜索需求选择合适的索引组件。
#
# *   **Agent (代理):**
#     *   **Tools:**  代理可以使用的外部工具，例如搜索引擎、计算器、数据库查询工具。
#     *   **Agent Types:**  不同的代理类型，例如 "zero-shot-react-description" (根据工具描述选择工具) 和 "conversational-react-description" (支持对话历史记录)。
#     *   **选择标准:**  根据任务的复杂性和所需的自动化程度选择合适的代理类型和工具。
#
# *   **Memory (记忆):**
#     *   **ConversationBufferMemory:**  存储完整的对话历史记录。
#     *   **ConversationSummaryMemory:**  总结对话历史记录以节省空间。
#     *   **ConversationBufferWindowMemory:**  只存储最近的对话轮次。
#     *   **选择标准:**  根据对话的长度和所需的上下文信息选择合适的记忆类型。
#
# **4.  构建你的 Langchain 应用:**
#
# 以下是一个使用 Langchain 构建简单问答系统的示例：
#
# ```python
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.llms import OpenAI
#
# # 1. 加载文档
# loader = TextLoader("your_document.txt")  # 替换为你的文档路径
# documents = loader.load()
#
# # 2. 分割文本
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
#
# # 3. 创建向量嵌入
# embeddings = OpenAIEmbeddings()  # 需要 OpenAI API 密钥
# db = Chroma.from_documents(texts, embeddings)
#
# # 4. 创建问答链
# llm = OpenAI(temperature=0)  # temperature 控制生成文本的随机性
# chain = load_qa_with_sources_chain(llm, chain_type="stuff")  # "stuff" 将所有相关文档填充到提示中
#
# # 5. 回答问题
# query = "What is the main topic of this document?"
# docs = db.similarity_search(query)
# result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
#
# print(result["output_text"])
# ```
#
# **解释:**
#
# *   **加载文档:** 使用 `TextLoader` 加载文本文件。
# *   **分割文本:** 使用 `CharacterTextSplitter` 将文档分割成更小的块。
# *   **创建向量嵌入:** 使用 `OpenAIEmbeddings` 将文本块转换为向量嵌入，并存储在 `Chroma` 向量存储中。
# *   **创建问答链:** 使用 `load_qa_with_sources_chain` 创建一个问答链，该链使用 `OpenAI` 模型和 "stuff" 链类型。
# *   **回答问题:** 使用 `similarity_search` 从向量存储中检索与查询相关的文档，并使用问答链来回答问题。
#
# **5.  部署和监控你的应用:**
#
# *   **部署:**  将你的 Langchain 应用部署到云平台（如 AWS, Azure, GCP）或本地服务器。
# *   **监控:**  监控应用的性能和错误，并进行必要的调整。
# *   **持续改进:**  根据用户反馈和数据分析，不断改进你的应用。
#
# **6.  最佳实践:**
#
# *   **提示工程:**  花时间构建有效的提示，以获得高质量的结果。  尝试不同的提示策略，并评估其对模型输出的影响。
# *   **数据质量:**  确保你的数据质量高，以避免模型产生错误或偏差。
# *   **模型选择:**  根据你的特定需求选择最佳模型。 考虑模型的性能、成本、速度和特定领域知识。
# *   **测试和评估:**  对你的应用进行彻底的测试和评估，以确保其正常工作。
# *   **安全性:**  采取适当的安全措施，以保护你的应用免受攻击。  例如，验证用户输入，防止提示注入攻击。
# *   **可扩展性:**  构建可扩展的应用，以便能够处理不断增长的数据量和用户流量。
# *   **模块化:**  将你的应用分解成更小的模块，以便于维护和更新。
# *   **版本控制:**  使用版本控制系统（如 Git）来管理你的代码。
# *   **文档化:**  编写清晰的文档，以便于他人理解和使用你的应用。
#
# **7.  进一步学习:**
#
# *   **Langchain 官方文档:** [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)
# *   **Langchain 社区:**  加入 Langchain 社区，与其他开发者交流经验和寻求帮助。
# *   **在线课程和教程:**  学习在线课程和教程，以深入了解 Langchain 的各个方面。
#
# **总结:**
#
# Langchain 提供了一个强大的框架，可以简化使用 LLMs 构建智能应用的过程。  通过理解 Langchain 的核心概念、选择合适的组件、遵循最佳实践，你可以构建各种创新的应用，例如问答系统、文档摘要、聊天机器人、代码生成等等。  记住，持续学习和实践是掌握 Langchain 的关键。  希望这个详细的解决方案能够帮助你开始你的 Langchain 之旅!


