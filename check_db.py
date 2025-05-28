from rag import engine, text

def search_similar_departments(university=None, department=None):
    with engine.connect() as conn:
        query = """
        SELECT id, university, department 
        FROM department_features 
        WHERE 1=1
        """
        params = {}
        
        if university:
            query += " AND university LIKE :uni"
            params['uni'] = f'%{university}%'
        
        if department:
            query += " AND department LIKE :dept"
            params['dept'] = f'%{department}%'
        
        query += " ORDER BY university, department"
        
        result = conn.execute(text(query), params).fetchall()
        
        print(f"\n找到 {len(result)} 個相關系所：")
        for row in result:
            print(f"ID: {row[0]} - {row[1]} {row[2]}")

if __name__ == "__main__":
    print("搜尋中央大學資管系相關系所...")
    search_similar_departments("中央", "資訊") 