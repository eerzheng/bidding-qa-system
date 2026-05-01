# Bidding QA System - 招投标智能问答系统

基于 RAG (检索增强生成) 与大模型技术的垂直领域问答系统。

## 🚀 核心功能
- **智能问答**：基于自然语言理解，精准回答招投标政策与流程问题。
- **语义检索**：利用向量数据库(FAISS)进行深度语义匹配，召回相关度高。
- **长链推理**：结合上下文进行多步逻辑推演，确保答案符合行业规范。

## 🛠️ 技术栈
- **开发框架**：Python, LangChain
- **AI 模型**：Sentence-Transformers (Embedding), OpenAI/Gemini (LLM)
- **数据存储**：FAISS (向量数据库)
- **部署 Demo**：Gradio (Hugging Face Spaces)

## 📂 文件结构
- `app.py`: Gradio 交互式 Web 界面入口
- `my_rag.py`: 核心 RAG 业务逻辑代码
- `requirements.txt`: 项目依赖库列表

## 💡 在线演示
👉 [点击访问 Hugging Face Spaces Demo](https://huggingface.co/spaces/你的用户名/bidding-qa-demo) 

*注：由于免费算力限制，首次加载可能需要 1-2 分钟，请耐心等待。*

---
开发者：eerzheng
