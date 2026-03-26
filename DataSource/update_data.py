import pandas as pd
import requests
from io import StringIO

print("🔧 กำลังเริ่มกระบวนการ FINAL UPDATE (กู้คืนข้อมูล 6 ปี)...")
print("========================================================")

# ลิงก์ข้อมูล 6 ปี
urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv", # 25/26
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv", # 24/25
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv", # 23/24
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv", # 22/23
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv", # 21/22
    "https://www.football-data.co.uk/mmz4281/2021/E0.csv"  # 20/21
]

dfs = []

for url in urls:
    try:
        # ดึงชื่อปีจาก URL เพื่อแสดงผล
        season = url.split('/')[-2]
        print(f"📂 กำลังโหลดฤดูกาล {season}...", end=" ")
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            # อ่าน CSV
            csv_data = StringIO(response.content.decode('latin-1'))
            df = pd.read_csv(csv_data)
            
            # ลบช่องว่างในชื่อคอลัมน์ (แก้ปัญหา 'Date ' vs 'Date')
            df.columns = df.columns.str.strip()
            
            if 'Date' in df.columns:
                # ---------------------------------------------------
                # 🛠️ ส่วนสำคัญ: ตัวแปลงวันที่อัจฉริยะ (Hybrid Parser)
                # ---------------------------------------------------
                # 1. ลองแปลงแบบอัตโนมัติ (dayfirst=True)
                # วิธีนี้จะจัดการทั้ง 19/09/20 และ 19/09/2020 ได้พร้อมกันส่วนใหญ่
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                
                # ลบแถวที่วันที่พังจริงๆ (เช่น แถวว่างท้ายไฟล์)
                df = df.dropna(subset=['Date'])
                
                # คัดเฉพาะคอลัมน์ที่ต้องใช้
                cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'AvgH', 'AvgD', 'AvgA']
                available_cols = [c for c in cols if c in df.columns]
                
                if len(available_cols) > 5:
                    df = df[available_cols]
                    dfs.append(df)
                    print(f"✅ สำเร็จ! (ได้ {len(df)} แมตช์)")
                else:
                    print("⚠️ ข้าม (คอลัมน์ไม่ครบ)")
            else:
                print("❌ ไม่พบคอลัมน์ Date")
        else:
            print(f"❌ โหลดไม่ได้ (HTTP {response.status_code})")
            
    except Exception as e:
        print(f"⚠️ Error: {e}")

# รวมร่าง
if dfs:
    print("\n🧩 กำลังรวมข้อมูลทั้งหมด...")
    full_data = pd.concat(dfs, ignore_index=True)
    full_data = full_data.sort_values(by='Date').reset_index(drop=True)
    
    # บันทึก
    full_data.to_csv('clean_premier_league.csv', index=False)
    
    print("\n" + "="*50)
    print(f"📊 สรุปยอดรวมสุดท้าย: {len(full_data)} แมตช์")
    print(f"📅 เริ่มต้น: {full_data.iloc[0]['Date'].date()}")
    print(f"📅 สิ้นสุด: {full_data.iloc[-1]['Date'].date()}")
    print("="*50)
    
    if len(full_data) > 2000:
        print("🎉 สำเร็จ! ข้อมูลครบ 6 ปีแล้ว")
        print("👉 ขั้นตอนต่อไป: รัน 'train_god_model.py' ได้เลย!")
    else:
        print("⚠️ ยอดยังน้อยไป (เช็คว่าเน็ตหลุดตอนโหลดบางไฟล์ไหม)")

else:
    print("❌ ไม่ได้ข้อมูลเลย")