# 模型配置（Ollama版本，CPU模式）
MODEL_CONFIG = {
    "ollama_model": "deepseek-r1:1.5b",  # Ollama中部署的LLM模型名称
    "api_url": "http://localhost:11434",  # Ollama服务地址
    "embedding_model": "BAAI/bge-small-zh-v1.5",  # 正确使用BGE-small版本
    "device": "cpu",  # 明确使用CPU
    "temperature": 0.3,
    "max_new_tokens": 512
}

# 检索配置（不变）
RETRIEVAL_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 3
}

# 数据配置（需用户自行替换路径）
DATA_CONFIG = {
    "doc_path": "zhihu_3k_rlhf_train.jsonl",  # 你的JSONL文件路径（相对路径）
    "vector_db_path": "./vector_db"
}
