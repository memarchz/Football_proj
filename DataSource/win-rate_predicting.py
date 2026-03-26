import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO

# =========================================================
# 1. โหลดความจำ (Database) เพื่อเอามาคำนวณ Form/H2H
# =========================================================
print("📂 กำลังโหลดฐานข้อมูลประวัติย้อนหลัง (เพื่อคำนวณฟอร์ม)...")

urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv", # ปีปัจจุบัน
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
]

dfs = []
for url in urls:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            csv_data = StringIO(response.content.decode('latin-1'))
            df = pd.read_csv(csv_data)
            df.columns = df.columns.str.strip()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                valid_cols = [c for c in cols if c in df.columns]
                dfs.append(df[valid_cols])
    except: pass

full_data = pd.concat(dfs, ignore_index=True)
full_data = full_data.sort_values(by='Date').reset_index(drop=True)
print(f"✅ โหลดข้อมูลครบ {len(full_data)} แมตช์")

# โหลดโมเดล
try:
    model = joblib.load('model_v3_specific.pkl')
    print("🧠 โหลดสมอง AI (V3.1) เรียบร้อย!")
except:
    print("❌ ไม่พบไฟล์โมเดล! กรุณารัน Part 1 ก่อน")
    exit()

# =========================================================
# 2. ฟังก์ชันคำนวณพลัง (Logic เดียวกับตอนเทรนเป๊ะๆ)
# =========================================================
def calculate_features(home_team, away_team, odds_h, odds_d, odds_a, history_df):
    
    # 1. คำนวณ xG (10 นัดล่าสุด)
    avg_h = history_df['FTHG'].mean()
    avg_a = history_df['FTAG'].mean()
    
    h_games = history_df[history_df['HomeTeam'] == home_team].tail(10)
    a_games = history_df[history_df['AwayTeam'] == away_team].tail(10)
    
    if len(h_games) < 3 or len(a_games) < 3:
        # กรณีหาไม่เจอ ให้ค่ากลางๆ
        h_xg, a_xg = 1.35, 1.35 
    else:
        h_att = h_games['FTHG'].mean() / avg_h
        h_def = h_games['FTAG'].mean() / avg_a
        a_att = a_games['FTAG'].mean() / avg_a
        a_def = a_games['FTHG'].mean() / avg_h
        h_xg = h_att * a_def * avg_h
        a_xg = a_att * h_def * avg_a

    # 2. คำนวณ H2H (6 นัดล่าสุด - ไม่สนบ้านใคร)
    h2h_matches = history_df[((history_df['HomeTeam'] == home_team) & (history_df['AwayTeam'] == away_team)) | 
                             ((history_df['HomeTeam'] == away_team) & (history_df['AwayTeam'] == home_team))].tail(6)
    
    h2h_points = 0
    if len(h2h_matches) > 0:
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team: # ทีมที่เราเชียร์เป็นเจ้าบ้าน
                if match['FTR'] == 'H': h2h_points += 3
                elif match['FTR'] == 'D': h2h_points += 1
            elif match['AwayTeam'] == home_team: # ทีมที่เราเชียร์เป็นทีมเยือน
                if match['FTR'] == 'A': h2h_points += 3
                elif match['FTR'] == 'D': h2h_points += 1
        h2h_score = h2h_points / len(h2h_matches)
    else:
        h2h_score = 1.0 # ไม่เคยเจอกัน ให้เสมอไว้ก่อน

    # 3. คำนวณ Specific Form (6 นัด)
    # เจ้าบ้าน: ฟอร์มในบ้าน 6 นัดหลัง
    home_sp_games = history_df[history_df['HomeTeam'] == home_team].tail(6)
    home_sp_points = 0
    for _, match in home_sp_games.iterrows():
        if match['FTR'] == 'H': home_sp_points += 3
        elif match['FTR'] == 'D': home_sp_points += 1
        
    # ทีมเยือน: ฟอร์มนอกบ้าน 6 นัดหลัง
    away_sp_games = history_df[history_df['AwayTeam'] == away_team].tail(6)
    away_sp_points = 0
    for _, match in away_sp_games.iterrows():
        if match['FTR'] == 'A': away_sp_points += 3
        elif match['FTR'] == 'D': away_sp_points += 1

    # สร้างข้อมูลส่งให้ AI
    return pd.DataFrame([[odds_h, odds_d, odds_a, h_xg, a_xg, h_xg-a_xg, h2h_score, home_sp_points, away_sp_points]], 
                        columns=['AvgH', 'AvgD', 'AvgA', 'Home_xG', 'Away_xG', 'xG_Diff', 'H2H_Avg_Points', 'Home_Home_Form', 'Away_Away_Form'])

# =========================================================
# 3. ✍️ โซนกรอกข้อมูลคู่ที่จะแข่ง (EDIT ตรงนี้!)
# =========================================================
# ใส่ชื่อทีม (ภาษาอังกฤษ) และ ราคา (Odds)
upcoming_matches = [
    # {"Home": "ชื่อเจ้าบ้าน", "Away": "ชื่อทีมเยือน", "Odds": [เจ้าบ้าน, เสมอ, ทีมเยือน]},
    
    {"Home": "Man United",    "Away": "Newcastle",     "Odds": [2.522, 3.72, 2.857]},
    {"Home": "Nott'm Forest", "Away": "Man City",      "Odds": [5.48, 4.365, 1.656]},
    {"Home": "Arsenal",       "Away": "Brighton",      "Odds": [1.464, 4.93, 7.70]},
    {"Home": "Brentford",     "Away": "Bournemouth",   "Odds": [2.338, 3.76, 3.11]},
    {"Home": "Burnley",       "Away": "Everton",       "Odds": [4.27, 3.515, 2.013]},
    {"Home": "Liverpool",     "Away": "Wolves",        "Odds": [1.26, 6.91, 12.90]},
    {"Home": "West Ham",      "Away": "Fulham",        "Odds": [2.796, 3.505, 2.684]},
    
    # คุณสามารถเพิ่มคู่ตรงนี้ได้เลย...
]

# =========================================================
# 4. ประมวลผลและทำนาย (แบบละเอียดเห็นไส้ใน)
# =========================================================
print(f"\n{'='*100}")
print(f"{'MATCH':<30} | {'HOME %':<8} | {'DRAW %':<8} | {'AWAY %':<8} | {'PREDICTION':<12}")
print(f"{'='*100}")

for match in upcoming_matches:
    home = match['Home']
    away = match['Away']
    odds = match['Odds']
    
    # สร้าง Features
    input_data = calculate_features(home, away, odds[0], odds[1], odds[2], full_data)
    
    # ทำนาย
    proba = model.predict_proba(input_data)[0] # ได้ค่า [Away%, Draw%, Home%]
    
    # เรียงลำดับค่าความน่าจะเป็น (Model เรียงตามตัวอักษร A, D, H)
    away_prob = proba[0] * 100
    draw_prob = proba[1] * 100
    home_prob = proba[2] * 100
    
    # หาตัวที่มั่นใจที่สุด
    max_prob = max(away_prob, draw_prob, home_prob)
    if max_prob == home_prob: pred = "Home 🏠"
    elif max_prob == away_prob: pred = "Away ✈️"
    else: pred = "Draw 🤝"

    # Action Marker
    marker = ""
    if max_prob >= 75: marker = "✅ PLAY"
    if max_prob >= 80: marker = "💎 DIAMOND"
    
    print(f"{home} vs {away:<15} | {home_prob:.1f}%   | {draw_prob:.1f}%   | {away_prob:.1f}%   | {pred:<10} {marker}")

print(f"{'='*100}")