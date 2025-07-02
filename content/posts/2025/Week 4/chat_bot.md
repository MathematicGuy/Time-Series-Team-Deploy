---
title: "Tuần 4 - Trợ lý AI đã xuất hiện"
date: 2025-06-28T13:03:07+07:00
description: Trợ lý AI tiếng Việt hỗ trợ hỏi đáp từ tài liệu PDF bằng công nghệ RAG kết hợp mô hình Vicuna-7B, được xây dựng bằng Streamlit và LangChain.
image: images/nasa-Ed2AELHKYBw-unsplash.jpg
caption: Photo by Nasa on Unsplash
categories:
  - feature
tags:
  - feature
draft: false
---

## 🤖 Trợ Lý AI Tiếng Việt — PDF RAG Assistant

Chào mừng bạn đến với chatbot AI thông minh của nhóm, được huấn luyện để **trả lời câu hỏi từ tài liệu PDF** bằng tiếng Việt.

👉 **Bạn có thể hỏi:**

- Với nội dung cho cá nhân: Bạn có thể tải lên một văn bản hoặc đường dẫn tiếng việt và đặt câu hỏi xung quanh tài liệu đó, Trợ lý AI sẽ giúp bạn đưa ra thông tin liên quan
- Với nội dung lớp AIO từ Tuần 1 đến giờ: Bạn chọn phần Git Respository, tại đó nhóm có đặt link mặc định đến blog kiến thức tổng hợp của lớp và bạn có thể đặt câu hỏi để Trợ lý AI có thể giúp bạn ôn lại kiến thức liên quan AIO
---

### 🧠 Cách thức hoạt động
Chatbot được xây dựng bằng:
- **Streamlit** để tạo giao diện đơn giản, dễ dùng
- **Langchain + HuggingFace** để hiểu ngữ cảnh và tạo câu trả lời
- **RAG (Retrieval-Augmented Generation)** để kết hợp nội dung từ PDF với mô hình ngôn ngữ

---

### 🛠️ Công nghệ sử dụng

| Thành phần | Công cụ |
|------------|---------|
| 🧪 Trải nghiệm Chatbot | [Streamlit](https://ragchatbotaio.streamlit.app/) |
| Code | [Google Colab](https://colab.research.google.com/drive/1RIqEgrFcSYTO6rlUj1jLoUuJFrtpZy4X?usp=sharing) |
| NLP model  | [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| Embedding tiếng Việt | `bkai-foundation-models/vietnamese-bi-encoder` |
| Xử lý PDF  | LangChain `PyPDFLoader` |
| Semantic Split | LangChain `SemanticChunker` |
| Truy xuất văn bản | ChromaDB |
| Truy vấn ngữ cảnh | RAG pipeline |
