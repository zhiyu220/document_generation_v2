import os
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF

# ==== 載入環境變數 ====
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ==== 初始化 ====
engine = create_engine(DATABASE_URL)
log_file = f"update_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ==== Tesseract 設定 ====
if os.name == 'nt':  # Windows 系統
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r'C:\poppler\poppler-24.08.0\Library\bin'  # 請根據實際安裝路徑修改
else:  # Linux/Mac 系統
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    POPPLER_PATH = None

# ==== OCR 相關函數 ====
def process_image(image_data):
    """處理圖片並進行OCR"""
    try:
        # 將圖片數據轉換為PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # 轉換為OpenCV格式進行預處理
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 圖片預處理
        # 1. 轉換為灰度圖
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 2. 自適應二值化
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # 3. 降噪
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # 4. 邊緣增強
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 轉回PIL Image格式
        processed_image = Image.fromarray(sharpened)
        
        # 進行OCR
        text = pytesseract.image_to_string(processed_image, lang='chi_tra+eng')
        return text.strip()
    except Exception as e:
        print(f"❌ 圖片處理失敗: {e}")
        return ""

def process_pdf(pdf_bytes):
    """處理PDF文件並進行OCR"""
    try:
        # 將PDF轉換為圖片
        if os.name == 'nt':
            images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
        else:
            images = convert_from_bytes(pdf_bytes)
        
        # 對每一頁進行OCR
        text_results = []
        for image in images:
            try:
                text = pytesseract.image_to_string(image, lang='chi_tra+eng')
                if text.strip():
                    text_results.append(text.strip())
            except Exception as e:
                print(f"OCR處理單頁時發生錯誤: {str(e)}")
                continue
        
        return '\n\n'.join(text_results)
    except Exception as e:
        print(f"❌ PDF處理失敗: {e}")
        return ""

def extract_text_from_file(file_path):
    """從文件中提取文字（支援PDF和圖片）"""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            # 先嘗試直接提取PDF文字
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # 如果直接提取的文字太少，使用OCR
            if len(text.strip()) < 100:
                with open(file_path, 'rb') as f:
                    text = process_pdf(f.read())
            return text.strip()
            
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            with open(file_path, 'rb') as f:
                return process_image(f.read())
        
        else:
            raise ValueError(f"不支援的文件格式: {file_extension}")
            
    except Exception as e:
        print(f"❌ 文件處理失敗: {e}")
        return ""

# ==== 載入向量資料庫 ====
vectorstore = FAISS.load_local(
    "db/faiss_store",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

def clean_markdown(text: str) -> str:
    # 移除 Markdown 標題符號 (#)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # 移除粗體與斜體標記（**text**、*text*、_text_）
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    # 移除項目前綴符號（- 或 * 或數字.）
    text = re.sub(r"^\s*([-*]|[0-9]+\.)\s+", "", text, flags=re.MULTILINE)

    # 移除多餘空白行
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()

retriever = vectorstore.as_retriever(search_type="similarity", k=4)
model = genai.GenerativeModel("gemini-2.0-flash")

def find_similar_departments(conn, university, department):
    """搜尋相似的系所"""
    query = """
    SELECT id, university, department 
    FROM department_features 
    WHERE (university LIKE :uni_pattern OR university = :university)
    AND (department LIKE :dept_pattern OR department = :department)
    ORDER BY 
        CASE 
            WHEN university = :university AND department = :department THEN 0
            WHEN university = :university THEN 1
            WHEN department = :department THEN 2
            ELSE 3
        END,
        university, 
        department
    LIMIT 5
    """
    
    # 構建模糊搜尋的pattern
    uni_pattern = f"%{university}%"
    dept_pattern = f"%{department}%"
    
    results = conn.execute(text(query), {
        "university": university,
        "department": department,
        "uni_pattern": uni_pattern,
        "dept_pattern": dept_pattern
    }).fetchall()
    
    return results

def check_department_exists(conn, university, department):
    """檢查系所是否存在於資料庫中，如果不存在則搜尋相似系所"""
    # 先精確搜尋
    exact_match = conn.execute(text("""
        SELECT id FROM department_features 
        WHERE university = :university AND department = :department
    """), {
        "university": university,
        "department": department
    }).fetchone()
    
    if exact_match:
        return exact_match[0], None
    
    # 如果沒有精確匹配，搜尋相似系所
    similar_results = find_similar_departments(conn, university, department)
    if similar_results:
        print("\n💡 找到以下相似的系所：")
        for i, (id, uni, dept) in enumerate(similar_results, 1):
            print(f"{i}. {uni} {dept}")
        
        print("\n請選擇要使用的系所編號（輸入數字），或直接按 Enter 跳過：")
        choice = input().strip()
        
        if choice and choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(similar_results):
                return similar_results[choice_idx][0], None
    
    return None, None

def process_department_data(file_path, university=None, department=None):
    """處理單個系所的資料"""
    try:
        # 如果沒有提供學校和系所名稱，從檔名解析
        if not university or not department:
            filename = os.path.basename(file_path)
            name_parts = filename.replace('.pdf', '').split('_')
            if len(name_parts) >= 2:
                university = name_parts[0]
                # 移除檔案副檔名和可能的其他後綴
                department = '_'.join(name_parts[1:]).split('.')[0]
            else:
                raise ValueError(f"無法從檔名 {filename} 解析學校和系所資訊")

        print(f"🔍 正在處理：{university} {department}")
        
        # 先檢查系所是否存在
        with engine.begin() as conn:
            department_id, similar_departments = check_department_exists(conn, university, department)
            
            if not department_id:
                print(f"⚠️ 系所不存在於資料庫中：{university} {department}")
                return None

        # 提取文字
        extracted_text = extract_text_from_file(file_path)
        if not extracted_text:
            raise ValueError("無法從文件中提取文字")

        # 獲取相關文件
        base_prompt = f"{university}{department}"
        print(f"📚 搜尋相關資料：{base_prompt}")
        
        retrieved_docs = retriever.get_relevant_documents(f"{base_prompt}的備審建議與特色")
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

        # 合併提取的文字和相關文件
        combined_context = f"""
從文件中提取的內容：
{extracted_text}

相關系所資料：
{context}
"""

        print("🤖 生成系所特色...")
        feature_prompt = f"""請根據以下資料，整理「{university}{department}」的學系特色與培養重點，請使用純文字輸出，不要有markdown語法，格式固定如下，不要加入任何開場白或總結。每一點不超過 80 字，每點之間換行，表達要簡潔明瞭。
        一、學系特色：
        （請條列式寫出本系特色）

        二、培養重點：
        （請條列式寫出本系重視的培養能力或學生特質）

        以下是參考資料：
        {combined_context}
        """

        print("🤖 生成備審建議...")
        suggestion_prompt = f"""請根據以下資料，針對「{university}{department}」提供備審資料撰寫建議，使用純文字輸出，格式固定如下，不要加入開場白或總結。每一點建議不超過 80 字，每點之間換行，請簡潔明確。
        一、適合強調的能力或特質：
        （條列式列出建議呈現的能力與對應項目）

        二、建議呈現方式與撰寫方向：
        （條列式列出建議從哪些經歷或項目呈現，並簡要說明原因）

        三、可避免的常見問題：
        （條列式提醒常見錯誤或撰寫偏誤）

        以下是參考資料：
        {combined_context}
        """

        feature_ans_raw = model.generate_content(feature_prompt).text.strip()
        suggestion_ans_raw = model.generate_content(suggestion_prompt).text.strip()
        
        feature_ans = clean_markdown(feature_ans_raw)
        suggestion_ans = clean_markdown(suggestion_ans_raw)

        print("\n✨ 生成結果：")
        print("\n=== 學系特色 ===")
        print(feature_ans)
        print("\n=== 備審建議 ===")
        print(suggestion_ans)

        # 更新資料庫
        with engine.begin() as conn:
            print(f"\n📝 更新資料庫...")
            update_result = conn.execute(text("""
                UPDATE department_features 
                SET features = :features, 
                    suggestion_content = :suggestion
                WHERE id = :id
                RETURNING id
            """), {
                "features": feature_ans,
                "suggestion": suggestion_ans,
                "id": department_id
            })
            
            if update_result.rowcount > 0:
                print(f"✅ 成功更新資料庫！")
                return {
                    "university": university,
                    "department": department,
                    "features": feature_ans,
                    "suggestion": suggestion_ans,
                    "extracted_text": extracted_text
                }
            else:
                print("❌ 資料庫更新失敗")
                return None

    except Exception as e:
        print(f"❌ 處理失敗 ({university}{department}): {e}")
        return None

def process_folder(folder_path):
    """處理資料夾中的所有文件"""
    processed_count = 0
    error_count = 0
    
    with engine.begin() as conn:
        with open(log_file, "w", encoding="utf-8") as log:
            log.write(f"📝 RAG 處理紀錄 {datetime.now()}\n\n")
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.bmp')):
                        file_path = os.path.join(root, file)
                        print(f"\n📄 處理文件：{file}")
                        
                        try:
                            result = process_department_data(file_path)
                            if not result:
                                error_count += 1
                                continue
                            
                            # 記錄日誌
                            log.write(f"✅ {result['university']}{result['department']}：\n")
                            log.write(f"[FEATURES]\n{result['features'][:200]}...\n")
                            log.write(f"[SUGGESTIONS]\n{result['suggestion'][:200]}...\n")
                            log.write(f"[EXTRACTED_TEXT]\n{result['extracted_text'][:200]}...\n\n")
                            
                            processed_count += 1
                            
                            print(f"✨ 成功處理：{result['university']}{result['department']}")
                            
                        except Exception as e:
                            error_count += 1
                            log.write(f"❌ 錯誤 ({file}): {str(e)}\n\n")
                            print(f"❌ 處理失敗 ({file}): {e}")
    
    print(f"\n📊 處理完成：")
    print(f"✅ 成功：{processed_count} 個文件")
    print(f"❌ 失敗：{error_count} 個文件")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        if os.path.isfile(target_path):
            try:
                result = process_department_data(target_path)
                if result:
                    print("\n是否要更新資料庫？(y/n)")
                    choice = input().strip().lower()
                    if choice == 'y':
                        with engine.begin() as conn:
                            # 更新資料庫
                            print(f"\n📝 更新資料庫...")
                            update_result = conn.execute(text("""
                                INSERT INTO department_features (university, department, features, suggestion_content)
                                VALUES (:university, :department, :features, :suggestion)
                                ON CONFLICT (university, department)
                                DO UPDATE SET
                                    features = EXCLUDED.features,
                                    suggestion_content = EXCLUDED.suggestion_content
                            """), result)
                            print("✅ 成功更新資料庫！")
            except Exception as e:
                print(f"❌ 處理檔案時發生錯誤：{str(e)}")
        elif os.path.isdir(target_path):
            print(f"🔍 開始處理目錄：{target_path}")
            total_files = sum([len(files) for _, _, files in os.walk(target_path) if any(f.endswith('.pdf') for f in files)])
            processed = 0
            success = 0
            errors = []
            
            print(f"\n📊 總共發現 {total_files} 個 PDF 檔案")
            
            for root, _, files in os.walk(target_path):
                for file in files:
                    if file.endswith('.pdf'):
                        file_path = os.path.join(root, file)
                        processed += 1
                        print(f"\n📄 處理檔案 ({processed}/{total_files}): {file}")
                        
                        try:
                            result = process_department_data(file_path)
                            if result:
                                with engine.begin() as conn:
                                    # 更新資料庫
                                    print(f"\n📝 更新資料庫...")
                                    update_result = conn.execute(text("""
                                        INSERT INTO department_features (university, department, features, suggestion_content)
                                        VALUES (:university, :department, :features, :suggestion)
                                        ON CONFLICT (university, department)
                                        DO UPDATE SET
                                            features = EXCLUDED.features,
                                            suggestion_content = EXCLUDED.suggestion_content
                                    """), result)
                                    print("✅ 成功更新資料庫！")
                                    success += 1
                            else:
                                error_msg = f"❌ 處理 {file} 失敗：無法生成內容"
                                print(error_msg)
                                errors.append(error_msg)
                        except Exception as e:
                            error_msg = f"❌ 處理 {file} 時發生錯誤：{str(e)}"
                            print(error_msg)
                            errors.append(error_msg)
            
            print("\n📊 處理完成統計：")
            print(f"總檔案數：{total_files}")
            print(f"成功處理：{success}")
            print(f"處理失敗：{len(errors)}")
            
            if errors:
                print("\n❌ 錯誤清單：")
                for error in errors:
                    print(error)
        else:
            print("❌ 指定的路徑不存在")
    else:
        print("❌ 請提供要處理的檔案或目錄路徑")
        print("用法：python rag.py <檔案路徑或目錄路徑>")
