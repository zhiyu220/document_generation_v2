import os
import re
import fitz  # PyMuPDF
import pdfplumber
import json
import psycopg2
from dotenv import load_dotenv
from collections import defaultdict
import google.generativeai as genai

# === 環境變數 ===
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# === SECTION_MAPPING ===
SECTION_MAPPING = {
    "A": "修課紀錄", "B": "書面報告", "C": "實作作品", "D": "自然科學探究",
    "E": "社會領域探究", "F": "高中自主學習計畫", "G": "社團活動經驗",
    "H": "幹部經驗", "I": "服務學習經驗", "J": "競賽表現",
    "K": "非修課成果作品", "L": "檢定證照", "M": "特殊優良表現",
    "N": "多元表現綜整心得", "O": "高中學習歷程反思", "P": "學習動機",
    "Q": "未來學習計畫", "R":"科系自訂",
    "S":"科系自訂",
    "T":"科系自訂"
}

# === 解析 PDF 表格 ===
def extract_and_clean_table(pdf_path):
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                cleaned_table = []
                for row in table:
                    # 過濾空欄位，並去除 None
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    if any(cleaned_row):  # 確保行內有內容
                        cleaned_table.append(cleaned_row)
                extracted_data.extend(cleaned_table)
    
    return extracted_data

def parse_three_column_table(table_data):
    structured_data = []
    current_section = None
    
    for row in table_data:
        if len(row) >= 3:  # 確保至少有 3 列（項目、審查重點、準備指引）
            section_code = None
            
            # 嘗試從第一列中提取 section_code
            for part in row[0].split("\n"):
                if "(" in part and ")" in part:
                    section_code = part[part.find("(")+1:part.find(")")]
                    break
            
            # 若提取到 section_code，則記錄該行
            if section_code:
                current_section = {
                    "section_code": section_code,
                    "section_name": row[0].split("\n")[0].strip(),
                    "content": f"審查重點: {row[1].strip()}。\n準備指引: {row[2].strip()}。"
                }
                structured_data.append(current_section)
            elif current_section:
                # 若當前行沒有新的 section_code，則認為是上一個 section 的補充內容
                current_section["content"] += f"\n{row[1].strip()}。\n{row[2].strip()}。"
    
    return structured_data

# === GPT 分析 ===
def analyze_with_gpt(text, tables, university, department):
    section_map_prompt = "\n".join([f'"{k}": "{v}"' for k, v in SECTION_MAPPING.items()])
    prompt = f"""
你是一個文檔分析專家，請根據以下 PDF 內容提取資料：

學校：{university if university != '未知' else '請從內容中判斷'}
學系：{department if department != '未知' else '請從內容中判斷'}
- **請根據 `section_map` 中的對應關係，幫助判斷 `section_code`**
- **可能沒有 `section_code`，請根據內容合理判斷，如果遇到R,S,T，請將科系提供之section name填入**
- **可能沒有按照順序排列，請整理成正確的結構**
- **不改變原始表格結構，如果在同一格中請抓取完整內容**

以下是 section_map：
```json
{{
{section_map_prompt}
}}
```

PDF 提取的表格內容如下：
{json.dumps(tables, ensure_ascii=False, indent=2)}

請輸出 JSON 格式：
{{
  "university": "學校名稱",
        "department": "學系名稱",
        "department_features":"學系特色"，若無找到請留空白,
        "sections": [
            {{
                "section_code": "",
                "section_name": "",
                "content": ""
            }},
            {{
                "section_code": "",
                "section_name": "",
                "content": ""
            }}
        ]
}}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([prompt, text])
    content = response.candidates[0].content.parts[0].text.strip()
    #content = response.choices[0].message.content.strip()
        
    # ✅ 清理 ```json 或 ``` 符號（這些會讓 json.loads 出錯）
    if content.startswith("```json"):
        content = content[7:].strip()  # 去掉 ```json
    elif content.startswith("```"):
        content = content[3:].strip()  # 去掉 ```（沒有 json）

    if content.endswith("```"):
        content = content[:-3].strip()  # 去掉結尾的 ```
        
    return content

# === 主流程 ===
def main(pdf_path):
    university = os.path.basename(pdf_path).split("_")[0]
    department = os.path.basename(pdf_path).replace(".pdf", "").split("_")[1] if "_" in pdf_path else "未知"

    table_data = extract_and_clean_table(pdf_path)
    parsed_sections = parse_three_column_table(table_data)
    gpt_output = analyze_with_gpt("", tables, university, department)
    try:
        parsed_data = json.loads(gpt_output)
    except json.JSONDecodeError:
        print("❌ GPT 回傳格式錯誤")
        print(gpt_output)
        return

    print(json.dumps(parsed_data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_pdf_path = r"D:\lab\個人申請備審資料AI輔導系統\PDF\慈濟大學\114申請入學準備指引_108092護理系.pdf"
    main(test_pdf_path)