import os
import streamlit as st
from dotenv import load_dotenv

# LangChain 核心组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ==================== 1. 页面配置 ====================
st.set_page_config(page_title="RAG 知识库问答", page_icon="📚")
st.title("📚 个人私有知识库问答 (RAG)")
st.caption("基于通义千问 + FAISS 向量检索的文档问答系统")

# ==================== 2. 初始化大模型与向量化模型 ====================
# 阿里云通义千问支持 OpenAI 格式
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.1
)

# 使用阿里云的 Embedding 模型进行向量化
# 使用阿里云的 Embedding 模型进行向量化
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v3",
    check_embedding_ctx_length=False
)

# ==================== 3. 侧边栏：上传文档并构建知识库 ====================
with st.sidebar:
    st.header("文档上传")
    uploaded_file = st.file_uploader("请上传 PDF 或 TXT 文件", type=["pdf", "txt"])

    if uploaded_file is not None:
        # 将上传的文件保存到本地临时目录
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("构建知识库"):
            with st.spinner("正在解析文档并构建向量库..."):
                # 1. 加载文档
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file_path)
                else:
                    loader = TextLoader(temp_file_path, encoding="utf-8")
                docs = loader.load()

                # 2. 文本切分 (Chunking)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                splits = text_splitter.split_documents(docs)

                # 3. 向量化并存入 FAISS
                vectorstore = FAISS.from_documents(splits, embeddings)

                # 存入 Session State，以便后续对话使用
                st.session_state.vectorstore = vectorstore
                st.success("知识库构建完成！可以开始提问了。")

                # 清理临时文件
                os.remove(temp_file_path)

# ==================== 4. 主界面：对话交互 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("基于文档提问..."):
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 检查是否构建了知识库
    if "vectorstore" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("请先在左侧上传文档并点击【构建知识库】！")
    else:
        with st.chat_message("assistant"):
            with st.spinner("正在检索并思考..."):
                # 3. 配置检索器和 Prompt
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

                system_prompt = (
                    "你是一个专业的问答助手。请严格根据以下检索到的上下文回答问题。\n"
                    "如果你不知道答案，就说不知道，不要编造。\n\n"
                    "上下文：\n{context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # 4. 组装 RAG 链
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # 5. 执行 RAG
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]

                # 6. 显示结果
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})