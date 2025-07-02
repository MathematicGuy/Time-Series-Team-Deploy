---
title: "Tuần 4 - Trợ lý AI đã xuất hiện"
date: 2025-06-28T13:03:07+07:00
description: Trợ lý AI tiếng Việt hỗ trợ hỏi đáp từ tài liệu PDF bằng công nghệ RAG kết hợp mô hình Vicuna-7B, được xây dựng bằng Streamlit và LangChain.
image: images/nasa-Ed2AELHKYBw-unsplash.jpg
caption: Photo by Nasa on Unsplash
categories:
  - feature
tags: ["chatbot", "streamlit", "PDF", "hỗ trợ"]
draft: false
---

## 🤖 Trợ Lý AI Tiếng Việt — PDF RAG Assistant

Chào mừng bạn đến với chatbot AI thông minh của nhóm, được huấn luyện để **trả lời câu hỏi từ tài liệu PDF** bằng tiếng Việt.

👉 **Bạn có thể hỏi:**

- Với nội dung cho cá nhân: Bạn có thể tải lên một văn bản hoặc đường dẫn tiếng việt và đặt câu hỏi xung quanh tài liệu đó, Trợ lý AI sẽ giúp bạn đưa ra thông tin liên quan
- Với nội dung lớp AIO từ Tuần 1 đến giờ: Bạn chọn phần Git Respository, tại đó nhóm có đặt link mặc định đến blog kiến thức tổng hợp của lớp và bạn có thể đặt câu hỏi để Trợ lý AI có thể giúp bạn ôn lại kiến thức liên quan AIO
---

### 🧠 Cách hoạt động
Chatbot được xây dựng bằng:
- **Streamlit** để tạo giao diện đơn giản, dễ dùng
- **Langchain + HuggingFace** để hiểu ngữ cảnh và tạo câu trả lời
- **RAG (Retrieval-Augmented Generation)** để kết hợp nội dung từ PDF với mô hình ngôn ngữ

---

## 🧪 Trải nghiệm Chatbot

<div style="display: flex; justify-content: center; padding: 2rem;">
  <iframe src="https://ragchatbotaio.streamlit.app/" 
          width="100%" 
          height="800" 
          style="max-width: 1200px; border: 2px solid #ddd; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
          frameborder="0">
    <p>Trình duyệt của bạn không hỗ trợ iframe. 
       <a href="https://ragchatbotaio.streamlit.app/" target="_blank">Nhấn vào đây để mở chatbot</a>
    </p>
  </iframe>
</div>

> ⚠️ **Lưu ý:** Một số trình duyệt hoặc thiết lập bảo mật sẽ **chặn iframe**. Nếu chatbot không hiển thị, bạn có thể mở trực tiếp ở nút bên dưới:

---

### 🚀 Mở Chatbot trực tiếp

<div style="text-align: center; padding: 1rem;">
  <a href="https://ragchatbotaio.streamlit.app/" 
     target="_blank" 
     style="display: inline-block; background: linear-gradient(90deg, #006400, #009900); color: white; padding: 12px 24px; border-radius: 25px; text-decoration: none; font-weight: bold;">
    🇻🇳 Mở Chatbot Tiếng Việt
  </a>
</div>

---

### 🛠️ Công nghệ sử dụng

| Thành phần | Công cụ |
|------------|---------|
| Giao diện  | [Streamlit](https://ragchatbotaio.streamlit.app/) |
| NLP model  | [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| Embedding tiếng Việt | `bkai-foundation-models/vietnamese-bi-encoder` |
| Xử lý PDF  | LangChain `PyPDFLoader` |
| Semantic Split | LangChain `SemanticChunker` |
| Truy xuất văn bản | ChromaDB |
| Truy vấn ngữ cảnh | RAG pipeline |

---

## 📥 Cần hỗ trợ?

Nếu bạn muốn triển khai chatbot tương tự cho nhóm, lớp học, doanh nghiệp hay dự án cá nhân, hãy liên hệ nhóm để được hỗ trợ setup!

---

🧠 _Mọi câu hỏi đều có thể bắt đầu bằng một tệp PDF._
