import os
import fitz  # PyMuPDF
import pickle
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 讀取 .env 裡的 OPENAI_API_KEY
load_dotenv()
vectorstore = FAISS.load_local("db/faiss_store", OpenAIEmbeddings())

# 建立向量資料庫
def process_pdfs_to_vectorstore(pdf_folder: str, output_path: str):
    embeddings = OpenAIEmbeddings()
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for root, _, files in os.walk(pdf_folder):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                doc = fitz.open(pdf_path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()

                if not full_text.strip():
                    print(f"⚠️ {filename} 沒有有效文字，已略過")
                    continue

                chunks = splitter.split_text(full_text)
                if not chunks:
                    print(f"⚠️ {filename} 分段後為空，已略過")
                    continue

                name_parts = filename.replace(".pdf", "").split("_")
                school = name_parts[0] if len(name_parts) > 0 else "未知學校"
                department = name_parts[1] if len(name_parts) > 1 else "未知科系"

                metadata = {
                    "school": school,
                    "department": department,
                    "source_pdf": filename,
                    "source_path": os.path.relpath(pdf_path, pdf_folder)
                }

                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata=metadata))


    print(f"📄 共建立 {len(documents)} 段文字 chunk")

    if not documents:
        print("❌ 無任何有效資料，未建立向量庫")
        return

    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vectorstore.save_local(output_path)  # output_path 建議用資料夾名稱

    print(f"✅ 向量庫已儲存至 {output_path}")

# 執行
if __name__ == "__main__":
    process_pdfs_to_vectorstore("data", "db/faiss_store")
