import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
import shutil

# === 載入 API 金鑰 ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# === 遞迴取得所有 PDF 路徑 ===
def get_all_pdfs(root_folder):
    pdf_paths = []
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(foldername, filename))
    return pdf_paths

# === 從 PDF 擷取前幾頁文字 ===
def extract_pdf_text(pdf_path, max_pages=3):
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(max_pages, len(doc))):
        text += doc[i].get_text()
    return text.strip()

# === 使用 GPT 推測檔名 ===
def generate_filename_from_text(text):
    prompt = f"""以下是大學備審 PDF 的部分文字內容，請幫我判斷這是哪個學校的哪個科系，並用「學校_科系」格式回覆，不需要附加解釋。例如：「元智大學_資訊管理學系」。

文字如下：
{text[:1500]}"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    filename = response.text.strip().replace(" ", "").replace("/", "_")
    return filename

# === 主流程 ===
def rename_pdfs_in_folder(root_folder, output_folder="data"):
    os.makedirs(output_folder, exist_ok=True)
    pdf_paths = get_all_pdfs(root_folder)
    print(f"📁 共找到 {len(pdf_paths)} 份 PDF\n")

    for path in pdf_paths:
        try:
            text = extract_pdf_text(path)
            new_name = generate_filename_from_text(text)
            new_path = os.path.join(output_folder, new_name + ".pdf")

            shutil.copy2(path, new_path)  # 保留原始檔案，也可改成 os.rename()
            print(f"✅ 已重新命名：{path} → {new_path}")

        except Exception as e:
            print(f"❌ 無法處理 {path}：{e}")

# ✅ 執行範例
if __name__ == "__main__":
    rename_pdfs_in_folder("raw_pdfs")  # 你原始的資料夾位置
