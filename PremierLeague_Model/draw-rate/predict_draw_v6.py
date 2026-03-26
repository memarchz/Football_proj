import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO

# =========================================================
# 1. โหลดข้อมูล (เพื่อใช้ดึงประวัติเก่ามาคำนวณ)
# =========================================================
print("🎯 LOADING V8.1 SNIPER SCOPE...")
urls = ["https://www.football-data.co.uk/mmz4281/2526/E0.csv", 
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv"]
dfs = []
for url in urls:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.content.decode('latin-1')))
            df.columns = df.columns.str.strip()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                dfs.append(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']])
    except: pass
full_data = pd.concat(dfs).sort_values('Date').reset_index(drop=True)

# โหลดโมเดล
try:
    model = joblib.load('model_v8_draw_box.pkl')
    print("🔫 SNIPER MODEL LOADED: พร้อมล่า!")
except: 
    print("❌ ไม่เจอไฟล์โมเดล (ต้องรันไฟล์ Train ก่อน)")
    exit()

# =========================================================
# 2. ฟังก์ชันคำนวณ (เหมือนตอน Train เป๊ะๆ)
# =========================================================
def analyze_match(home, away, oh, od, oa):
    # --- 1. THE FILTER (ด่านคัดกรอง) ---
    # เงื่อนไข V8.1: ราคาต่างกัน < 1.2 และ ทีมต่อจ่าย > 2.00
    is_in_box = (abs(oh - oa) < 1.2) and (oh > 2.00) and (oa > 2.00)
    
    if not is_in_box:
        return None, "OUT OF RANGE (ไม่สูสีพอ)", 0

    # --- 2. Feature Extraction ---
    h_form = full_data[full_data['HomeTeam'] == home].tail(10)
    a_form = full_data[full_data['AwayTeam'] == away].tail(10)
    
    # ถ้าข้อมูลไม่พอ ให้ใส่ค่ากลาง (เหมือนตอนแก้บั๊ก)
    if len(h_form) < 1 or len(a_form) < 1:
        tg_avg, imp_draw, recent_habit = 2.5, 28.0, 0
    else:
        # Total Goals
        h_g = (h_form['FTHG'].mean() + h_form['FTAG'].mean())
        a_g = (a_form['FTHG'].mean() + a_form['FTAG'].mean())
        # Handle NaN
        h_g = h_g if not pd.isna(h_g) else 2.5
        a_g = a_g if not pd.isna(a_g) else 2.5
        tg_avg = (h_g + a_g) / 2
        
        # Recent Habit
        recent_habit = len(h_form[h_form['FTR']=='D']) + len(a_form[a_form['FTR']=='D'])

    # Imp Draw & Odds Diff
    imp_draw = (1/od) * 100
    odds_diff = abs(oh - oa)

    # Prepare Input
    inp = pd.DataFrame([[tg_avg, imp_draw, odds_diff, recent_habit]], 
                       columns=['Total_Goals_Avg', 'Imp_Draw', 'Odds_Diff', 'Recent_Draw_Habit'])
    
    # --- 3. Predict ---
    prob = model.predict_proba(inp)[0][1] * 100
    
    status = "WAIT"
    if prob >= 40: status = "⚠️ RISKY"
    if prob >= 50: status = "🎯 FIRE (ยิงเลย)"
    if prob >= 60: status = "💎 GOD TIER"
    
    return prob, status, odds_diff

# =========================================================
# ✍️ โซนใส่คู่บอล (ใส่ราคาจริงจากเว็บ)
# =========================================================
# ตัวอย่าง: คู่ที่ราคาเสมอจ่ายน้อยๆ (3.00-3.30) และราคาเหย้าเยือนเบียดกัน
matches = [
    # คู่ตัวอย่าง (สมมติ)
    {"Home": "Man United",    "Away": "Newcastle",     "Odds": [2.52, 3.72, 2.85]},
    {"Home": "Nott'm Forest", "Away": "Man City",      "Odds": [5.48, 4.36, 1.65]},
    {"Home": "Arsenal",       "Away": "Brighton",      "Odds": [1.46, 4.93, 7.70]},
    {"Home": "Liverpool",     "Away": "Wolves",        "Odds": [1.26, 6.91, 12.9]},
    {"Home": "Brentford",     "Away": "Bournemouth",   "Odds": [2.338, 3.76, 3.11]},
     {"Home": "Chelsea",     "Away": "Aston Villa",   "Odds": [1.81, 3.945, 4.775]},
]

print(f"\n{'='*100}")
print(f"{'MATCH':<25} | {'ODDS (H-D-A)':<15} | {'CONF %':<8} | {'STATUS':<20}")
print(f"{'='*100}")

hit_count = 0

for m in matches:
    prob, status, diff = analyze_match(m['Home'], m['Away'], m['Odds'][0], m['Odds'][1], m['Odds'][2])
    
    odds_str = f"{m['Odds'][0]}-{m['Odds'][1]}-{m['Odds'][2]}"
    
    if prob is not None:
        # Highlight สีสำหรับตัวเทพ (ใน Terminal บางตัวอาจไม่ติดสี แต่ดูข้อความเอา)
        if prob >= 50: 
            hit_count += 1
            print(f"🔥 {m['Home']} vs {m['Away']:<10} | {odds_str:<15} | {prob:.2f}%   | {status}")
        else:
            print(f"   {m['Home']} vs {m['Away']:<10} | {odds_str:<15} | {prob:.2f}%   | {status}")
    else:
        # ข้ามพวกที่ไม่เข้าเกณฑ์
        pass 
        # print(f"❌ {m['Home']} vs {m['Away']:<10} | {odds_str:<15} | -        | {status}")

print(f"{'='*100}")
if hit_count == 0:
    print("😴 วันนี้ไม่มีคู่ระดับ 'Sniper' (>50%) เลย... แนะนำให้นอนครับ (อย่าฝืนเล่น)")
else:
    print(f"🚨 เจอเป้าหมายระดับพระเจ้า {hit_count} คู่! เช็คความชัวร์แล้วจัดไปครับ!")