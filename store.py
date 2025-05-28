import os
import fitz  # PyMuPDF
import pdfplumber
import psycopg2
import openai
import json
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict
from docx import Document
import pytesseract
from PIL import Image
import io
import numpy as np
import cv2
import pandas as pd

# ==== 載入環境變數 ====
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 設定 Tesseract 路徑 (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 設定 Tesseract OCR 配置
custom_config = r'--oem 3 --psm 6 -l chi_tra+eng'

# 設定 OpenAI API
#openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# SECTION_MAP
SECTION_MAPPING = {
    "A": "修課紀錄",
    "B": "書面報告",
    "C": "實作作品",
    "D": "自然科學探究",
    "E": "社會領域探究",
    "F": "高中自主學習計畫",
    "G": "社團活動經驗",
    "H": "幹部經驗",
    "I": "服務學習經驗",
    "J": "競賽表現",
    "K": "非修課成果作品",
    "L": "檢定證照",
    "M": "特殊優良表現",
    "N": "多元表現綜整心得",
    "O": "高中學習歷程反思",
    "P": "學習動機",
    "Q": "未來學習計畫",
    "R":"科系自訂",
    "S":"科系自訂",
    "T":"科系自訂"
}

# ==== 連接 PostgreSQL ====
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ====找到資料夾中所有檔案====
def process_doc_folder(folder_path):
    all_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".pdf", ".docx")):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("❌ 沒有找到 PDF 或 DOCX 檔案")
        return

    print(f"📂 發現 {len(all_files)} 份檔案，開始解析...")

    for file_path in all_files:
        filename = os.path.basename(file_path)
        university, department = extract_university_department(filename)
        #university = "國立金門大學"
        print(f"\n🚀 正在解析: {filename}")

        if filename.lower().endswith(".pdf"):
            text = parse_pdf_with_ocr(file_path)
        elif filename.lower().endswith(".docx"):
            text = parse_docx_with_ocr(file_path)
        else:
            continue

        print("🤖 使用 GPT 分析內容...")
        parsed_data = analyze_with_gpt(text, university, department)

        if parsed_data is None:
            print("❌ GPT 分析失敗，跳過此檔案")
            continue

        try:
            parsed_data = json.loads(parsed_data)
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析錯誤: {e}")
            continue

        parsed_data["sections"] = merge_duplicate_sections(parsed_data["sections"])
        print(f"✅ 完成：{parsed_data['university']} - {parsed_data['department']}")

        sql_output = generate_sql(parsed_data)
        with open("parsed_data.sql", "a", encoding="utf-8") as f:
            f.write(sql_output + "\n")

        store_in_database(parsed_data)
        print("✅ 資料已寫入資料庫")
        
# ==== 解析 PDF 檔案 ====
def parse_docx_with_ocr(docx_path):
    """解析Word文件，包含文字和圖片內容"""
    doc = Document(docx_path)
    full_text = []
    
    # 提取文字
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    # 提取圖片並進行OCR
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            try:
                image_data = rel.target_part.blob
                ocr_text = process_image(image_data)
                if ocr_text:
                    full_text.append(f"\n[圖片文字內容]:\n{ocr_text}")
            except Exception as e:
                print(f"❌ Word圖片處理失敗: {e}")
    
    return "\n".join(full_text)

def parse_pdf_with_ocr(pdf_path):
    """解析PDF文件，包含文字、圖片和表格內容"""
    extracted_text = ""
    
    # 使用pdfplumber提取表格
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 提取表格
            tables = page.extract_tables()
            if tables:
                extracted_text += f"\n[第{page_num + 1}頁表格]\n"
                for table in tables:
                    df = pd.DataFrame(table)
                    extracted_text += df.to_string(index=False, header=False) + "\n\n"
    
    # 使用PyMuPDF提取文字和圖片
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 提取文字
        text = page.get_text()
        if text:
            extracted_text += text + "\n"
        
        # 提取圖片並進行OCR
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            
            # 處理圖片並進行OCR
            ocr_text = process_image(image_data)
            if ocr_text:
                extracted_text += f"\n[圖片文字內容 {page_num + 1}-{img_index + 1}]:\n{ocr_text}\n"
    
    doc.close()
    
    # 後處理：移除重複內容和整理格式
    lines = extracted_text.split('\n')
    unique_lines = []
    seen = set()
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line and cleaned_line not in seen:
            seen.add(cleaned_line)
            unique_lines.append(line)
    
    return '\n'.join(unique_lines)

# ==== 從檔名提取學校 & 學系 ====
def extract_university_department(filename):
    """ 從檔名解析學校與學系名稱 """
    base_name = os.path.basename(filename).replace(".pdf", "")
    parts = base_name.split("_")

    if len(parts) >= 2:
        university = parts[0].strip()
        department = parts[1].strip()
    else:
        university = "未知"
        department = "未知"

    return university, department

# ==== 使用 GPT 解析內容 ====
def generate_department_features(text, university, department):
    """使用 LLM 生成學系特色"""
    prompt = f"""
    你是一個專業的學系分析專家。請根據以下內容，分析並總結出該學系的特色：

    學校：{university}
    學系：{department}

    內容：
    {text}

    請從以下幾個面向分析並生成學系特色：
    1. 課程與學習重點
    2. 未來發展方向
    3. 所需能力與特質
    4. 特色優勢

    請用簡潔的文字描述，不要超過300字。格式要求：
    1. 不要使用編號或標題
    2. 以流暢的段落形式呈現
    3. 確保內容具體且有價值
    4. 避免空泛的描述
    """

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": 0.7}
        )
        
        if not response or not hasattr(response, "candidates"):
            raise ValueError("Gemini API 回應為空")
        
        features = response.candidates[0].content.parts[0].text.strip()
        return features if features else "無法生成學系特色"

    except Exception as e:
        print(f"❌ 學系特色生成失敗: {e}")
        return "無法生成學系特色"

def analyze_with_gpt(text, university, department):
    """使用 GPT 分析文件內容"""
    # 先生成學系特色
    department_features = generate_department_features(text, university, department)
    print(f"\n🎯 生成的學系特色：\n{department_features}\n")

    # 在提示中加入對表格結構的特別說明
    prompt = f"""
    你是一個專業的文檔分析工具。請根據以下內容提取並格式化為JSON格式：
    
    特別注意：
    1. 內容包含表格形式的審查重點項目，請正確解析表格中的對應關係
    2. 表格中的百分比權重請保留在section_name中
    3. 確保正確識別各個項目的對應關係
    4. 使用提供的學系特色
    
    學校名稱: {university if university != "未知" else "請從內容提取"}
    學系名稱: {department if department != "未知" else "請從內容提取"}
    學系特色: {department_features}
    
    內容：
    {text}
    
    請輸出標準JSON格式：
    {{
        "university": "學校名稱",
        "department": "學系名稱",
        "department_features": "學系特色",
        "sections": [
            {{
                "section_code": "",
                "section_name": "",
                "content": ""
            }}
        ]
    }}
    """
    
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(
            prompt, 
            generation_config={"temperature": 0.6}
        )
        
        if not response or not hasattr(response, "candidates"):
            raise ValueError("Gemini API 回應為空")
        
        content = response.candidates[0].content.parts[0].text.strip()
        
        if content.startswith("```json"):
            content = content[7:].strip()
        elif content.startswith("```"):
            content = content[3:].strip()

        if content.endswith("```"):
            content = content[:-3].strip()

        print("🔹 GPT 回應內容：")
        print(content)

        if not content:
            raise ValueError("GPT 回應為空")

        return content

    except Exception as e:
        print(f"❌ GPT 分析錯誤: {e}")
        return None

# ==== 合併多筆資料 ====
def merge_duplicate_sections(sections):
    merged = defaultdict(lambda: {"section_name": "", "content": ""})

    for section in sections:
        code = section["section_code"]
        name = section["section_name"]
        content = section["content"] if section["content"] is not None else ""  # 防止 None

        if merged[code]["section_name"] == "":
            merged[code]["section_name"] = name

        if merged[code]["content"]:
            merged[code]["content"] += "\n" + content.strip()
        else:
            merged[code]["content"] = content.strip()

    return [
        {"section_code": code, "section_name": value["section_name"], "content": value["content"]}
        for code, value in merged.items()
    ]
    
# ==== 生成 SQL 語法 ====
def generate_sql(data):
    sql_statements = []
    # **1️⃣ 先處理 `application_guidelines` 的 SQL**
    for section in data["sections"]:
        sql = f"""
        INSERT INTO application_guidelines (university, department, section_code, section_name, content)
        VALUES ('{data["university"]}', '{data["department"]}', '{section["section_code"]}', 
                '{section["section_name"]}', '{section["content"].replace("'", "''")}')
        ON CONFLICT (university, department, section_code) DO UPDATE
        SET section_name = EXCLUDED.section_name,
            content = EXCLUDED.content;
        """
        sql_statements.append(sql.strip())

    # **2️⃣ 新增 `department_features` 的 SQL**
    department_features_sql = f"""
    INSERT INTO department_features (university, department, features)
    VALUES ('{data["university"]}', '{data["department"]}', '{data.get("department_features", "未提供學系特色").replace("'", "''")}')
    ON CONFLICT (university, department) DO UPDATE
    SET features = EXCLUDED.features;
    """
    sql_statements.append(department_features_sql.strip())

    return "\n".join(sql_statements)

# ==== 儲存到資料庫 ====
def store_in_database(data):
    conn = get_db_connection()
    cursor = conn.cursor()

    # **儲存備審指引**
    for section in data["sections"]:
        cursor.execute("""
            INSERT INTO application_guidelines (university, department, section_code, section_name, content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (university, department, section_code) DO UPDATE
            SET section_name = EXCLUDED.section_name,
                content = EXCLUDED.content;
        """, (
            data["university"], data["department"],
            section["section_code"], section["section_name"],
            section["content"]
        ))
        
     # **儲存學系特色**
    cursor.execute("""
        INSERT INTO department_features (university, department, features)
        VALUES (%s, %s, %s)
        ON CONFLICT (university, department) DO UPDATE
        SET features = EXCLUDED.features;
    """, (
        data["university"], data["department"],
        data.get("department_features", "未提供學系特色")
    ))

    conn.commit()
    cursor.close()
    conn.close()

# ==== OCR 相關函數 ====
def extract_text_from_image(image):
    """從圖片中提取文字"""
    try:
        # 使用 pytesseract 進行 OCR，指定語言為繁體中文
        text = pytesseract.image_to_string(image, lang='chi_tra')
        return text.strip()
    except Exception as e:
        print(f"❌ OCR 處理失敗: {e}")
        return ""

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
        
        # 進行OCR，使用自定義配置
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        
        # 提取表格
        try:
            tables = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DATAFRAME, config=custom_config)
            # 過濾出有效的表格數據
            valid_tables = tables[tables['conf'] > 60]
            if not valid_tables.empty:
                text += "\n\n[表格內容]\n"
                for _, row in valid_tables.iterrows():
                    if str(row['text']).strip():
                        text += f"{row['text']} "
        except Exception as e:
            print(f"⚠️ 表格提取失敗: {e}")
        
        return text.strip()
    except Exception as e:
        print(f"❌ 圖片處理失敗: {e}")
        return ""

# ==== 主程式 ====
def main(pdf_path):
    
    university, department = extract_university_department(pdf_path)
    university ="中國文化大學"
    #department =""
    print(university, department)
    
    print("📂 解析 PDF 中...")
    text = parse_pdf_with_ocr(pdf_path)
    #print(text)

    print("🤖 使用 GPT 分析內容...")
    parsed_data = analyze_with_gpt(text, university, department)

    if parsed_data is None:
        print("❌ 解析失敗，請檢查 GPT API 或 PDF 內容")
        exit(1)
        
    try:
        parsed_data = json.loads(parsed_data)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析錯誤: {e}")
        print("⚠️ GPT 原始回應內容如下（可能格式不正確或不完整）:")
        print(parsed_data)
        return
    
    # 合併重複 section_code 的內容
    parsed_data["sections"] = merge_duplicate_sections(parsed_data["sections"])

    print(f"\n✅ 解析完成，學校：{parsed_data['university']}，學系：{parsed_data['department']}")
    print(f"📌 共解析 {len(parsed_data['sections'])} 筆資料：")

    # 顯示解析的內容
    for section in parsed_data["sections"]:
        print(f"\n📝 {section['section_code']} - {section['section_name']}")
        print(f"📖 內容: {section['content']}")

    # 產生 SQL
    sql_output = generate_sql(parsed_data)
    sql_file = "parsed_data.sql"

    with open(sql_file, "w", encoding="utf-8") as f:
        f.write(sql_output)

    print(f"\n✅ SQL 語法已存至 `{sql_file}`，可手動執行以驗證資料。")

    # 存入資料庫
    print("📡 儲存到 PostgreSQL...")
    store_in_database(parsed_data)
    print("✅ 資料成功存入 PostgreSQL！")

if __name__ == "__main__":
    folder_path = r"D:\lab\個人申請備審資料AI輔導系統\PDF\中國文化大學"  # 設定目標資料夾
    process_doc_folder(folder_path)