import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="AI 面试八股文陪练", page_icon="🧠")
st.title("🧠 AI 面试八股文陪练 Agent")

# ==================== 兼容本地和云端的密钥读取 ====================
dashscope_key = os.getenv("DASHSCOPE_API_KEY")
qweather_key = os.getenv("QWEATHER_API_KEY")

if not dashscope_key:
    try:
        dashscope_key = st.secrets["DASHSCOPE_API_KEY"]
        qweather_key = st.secrets["QWEATHER_API_KEY"]
    except Exception as e:
        st.error(f"云端未找到 API Key，请检查 Secrets 配置：{e}")

# ==================== 1. 初始化模型 ====================
llm = ChatOpenAI(
    api_key=dashscope_key,  # ← 改成了刚才获取的变量
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.3
)

embeddings = OpenAIEmbeddings(
    api_key=dashscope_key,  # ← 改成了刚才获取的变量
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v3",
    check_embedding_ctx_length=False
)

# ==================== 2. 准备八股文知识库 ====================
# 这里为了演示，我们直接内置几段八股文文本。
# 实际使用中，你可以把你的八股文写在一个 txt 文件里，用 TextLoader 加载
knowledge_text = """
1. 什么是大模型的 Token？
Token 是大模型处理文本的最小单位。模型无法直接理解文字，需要通过 Tokenizer 分词器将文本切分成 Token，再映射成数字 ID 喂给模型。

2. 大模型的幻觉是什么？如何解决？
大模型幻觉是指模型一本正经地胡说八道。因为大模型底层是概率预测而非事实检索。解决办法是使用 RAG 技术，检索真实文档作为上下文，并约束 Prompt。

3. LangChain 的 Agent 是什么？
Agent 是让大模型具备自主决策能力的框架。通过 Function Calling，大模型可以自主判断用户意图，调用外部工具（如查天气、算数），并整合结果返回。
"""

if "vectorstore" not in st.session_state:
    # 文本切片
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.create_documents([knowledge_text])
    # 存入向量库
    st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)

# ==================== 3. 构建带记忆的 RAG 链 ====================
retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})

# (1) 让大模型根据聊天记录，重写用户的问题（解决代词指代问题）
contextualize_q_system_prompt = (
    "根据聊天历史和最新问题，生成一个独立的问题。如果最新问题不依赖历史，直接返回原问题。"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# (2) 最终的问答 Prompt
qa_system_prompt = (
    "你是一个专业的 AI 面试官。请根据以下参考资料回答用户的问题。回答要详细、有条理。\n"
    "如果你不知道答案，就说不知道，不要编造。\n\n"
    "参考资料：\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ==================== 4. 前端聊天界面与记忆管理 ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 显示历史聊天
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 接收输入
if prompt := st.chat_input("向 AI 面试官提问（比如：什么是 Token？那它的缺点呢？）"):
    # 显示用户输入
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.chat_history.append(AIMessage(content=answer))