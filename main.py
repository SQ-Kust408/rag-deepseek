from config import MODEL_CONFIG, RETRIEVAL_CONFIG, DATA_CONFIG
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import os
import json
import logging
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 全局变量
embeddings = None
vector_db = None


# 初始化嵌入模型（CPU版本）
def initialize_embeddings():
    global embeddings
    device = MODEL_CONFIG["device"]
    logging.info(f"使用 {device} 设备初始化嵌入模型")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_CONFIG["embedding_model"],
        model_kwargs={"device": device}
    )


# 处理文档并生成/加载向量库（关键修复：添加 allow_dangerous_deserialization）
def process_or_load_vector_db():
    global vector_db
    vector_db_path = DATA_CONFIG["vector_db_path"]

    if not os.path.exists(vector_db_path):
        logging.info("未找到向量库，开始处理文档...")
        docs = load_and_split_documents()
        vector_db = FAISS.from_texts([doc.page_content for doc in docs], embeddings)
        vector_db.save_local(vector_db_path)
        logging.info(f"向量库已生成并保存至 {vector_db_path}")
    else:
        logging.info("加载现有向量库...")
        # 重要：允许危险反序列化（仅在信任向量库来源时使用！）
        vector_db = FAISS.load_local(
            vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True  # 必须添加此参数
        )
    return vector_db


# 加载并分块文档（增强数据校验）
def load_and_split_documents():
    doc_path = DATA_CONFIG["doc_path"]
    logging.info(f"开始加载文档：{doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        documents = []
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                documents.append(doc)
            except json.JSONDecodeError:
                logging.warning(f"跳过第 {line_num} 行无效JSON数据")

    texts = []
    for idx, doc in enumerate(documents):
        # 安全提取prompt和output（同前）
        prompt = str(doc.get("prompt", "")).strip() if isinstance(doc.get("prompt"), str) else ""
        outputs = [str(item).strip() for item in doc.get("output", []) if isinstance(item, str)]
        if prompt or outputs:
            texts.append(f"问题：{prompt}\n回答：\n" + "\n".join(outputs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RETRIEVAL_CONFIG["chunk_size"],
        chunk_overlap=RETRIEVAL_CONFIG["chunk_overlap"]
    )
    return text_splitter.create_documents(texts)  # 生成Document对象


# 生成回答（增强错误处理）
def generate_answer(question):
    global vector_db
    if vector_db is None:
        return "请先等待向量库初始化（首次加载可能需要时间）"

    try:
        # 语义检索
        relevant_docs = vector_db.similarity_search(question, k=RETRIEVAL_CONFIG["top_k"])
        if not relevant_docs:
            return "未找到相关上下文，请尝试更具体的问题"

        # 构建上下文
        context = "\n".join([f"[参考段落 {i + 1}]\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        prompt = f"""请根据以下提供的上下文，回答用户的问题。如果上下文未提及相关信息，可基于常识回答，但需说明“根据常识”。

上下文：
{context}

问题：
{question}

回答："""

        # 调用Ollama API
        payload = {
            "model": MODEL_CONFIG["ollama_model"],
            "prompt": prompt,
            "temperature": MODEL_CONFIG["temperature"],
            "max_new_tokens": MODEL_CONFIG["max_new_tokens"],  # 正确参数名
            "stream": False  # 禁用流式响应（确保返回完整JSON）
        }
        response = requests.post(
            f"{MODEL_CONFIG['api_url']}/api/generate",
            json=payload,
            timeout=30
        )
        # 打印原始响应（帮助调试）
        logging.info(f"API响应状态码：{response.status_code}")
        logging.debug(f"API原始响应：{response.text}")

        response.raise_for_status()
        result = response.json().get("response", "").strip()
        return result if result else "模型返回空回答"

    except requests.exceptions.HTTPError as e:
        return f"HTTP错误：{str(e)}，原始响应：{response.text[:200]}"  # 显示部分响应
    except json.JSONDecodeError as e:
        return f"JSON解析错误：{str(e)}，原始响应：{response.text[:200]}"
    except Exception as e:
        return f"处理错误：{str(e)}"

# 启动Gradio界面（添加加载状态提示和提交按钮）
def launch_interface():
    with gr.Blocks(title="本地化RAG问答系统（CPU版）") as interface:
        gr.Markdown("# 私有文档智能问答系统")
        gr.Markdown("支持文件上传（暂未启用）与对话历史管理")

        # 初始化状态（用于首次加载提示）
        with gr.Row():
            question_input = gr.Textbox(lines=3, label="请输入你的问题", interactive=False)
            answer_output = gr.Textbox(label="回答", interactive=False)
            submit_button = gr.Button("提交问题")  # 添加提交按钮

        # 加载向量库时禁用输入
        def enable_inputs():
            question_input.interactive = True
            answer_output.interactive = False
            submit_button.interactive = True
            return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)

        # 首次加载时显示加载状态
        gr.Examples(
            examples=["如何优化RAG系统？", "文档分块的最佳实践"],
            inputs=question_input,
            label="示例问题"
        )

        # 绑定生成函数到提交按钮
        submit_button.click(generate_answer, question_input, answer_output)

        # 初始化流程
        initialize_embeddings()
        process_or_load_vector_db()
        enable_inputs()  # 加载完成后启用输入

    interface.launch(debug=True)


if __name__ == "__main__":
    launch_interface()
