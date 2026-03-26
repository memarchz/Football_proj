import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO

# =========================================================
# 1. โหลดฐานข้อมูลประวัติ (ต้องใช้คำนวณค่าพลังปัจจุบัน)
# =========================================================
print("📂 กำลังโหลดฐานข้อมูลล่าสุดเพื่อวิเคราะห์ฟอร์ม...")

# ⚠️ เลือกชุด URL ให้ตรงกับลีกที่จะทาย (อันนี้เป็นของอังกฤษ)
# ถ้าจะทายเยอรมัน ให้เปลี่ยน URL เป็นกลุ่ม D1, D2
urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E1.csv",
    "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2526/E3.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E3.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E3.csv",
]

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
    model = joblib.load('model_v5_intervals.pkl')
    print("🧠 โหลดสมอง V5.0 เรียบร้อย พร้อมใช้งาน!")
except:
    print("❌ ไม่พบไฟล์โมเดล! (กรุณารัน Part Training ก่อน)"); exit()

# =========================================================
# 2. ฟังก์ชันสร้าง Input (Logic เดียวกับตอนเทรนเป๊ะๆ)
# =========================================================
def prepare_match_features(home, away, oh, od, oa, history):
    # 1. Market Wisdom
    imp_h, imp_a = (1/oh)*100, (1/oa)*100

    # 2. xG Stats (อิงจากประวัติทั้งหมดที่มี)
    avg_h, avg_a = history['FTHG'].mean(), history['FTAG'].mean()
    h_games = history[history['HomeTeam'] == home].tail(10)
    a_games = history[history['AwayTeam'] == away].tail(10)

    if len(h_games) < 3: xg_stats = [1.3, 1.3, 0] # ค่า Default กรณีทีมใหม่
    else:
        h_att = h_games['FTHG'].mean() / avg_h
        h_def = h_games['FTAG'].mean() / avg_a
        a_att = a_games['FTAG'].mean() / avg_a
        a_def = a_games['FTHG'].mean() / avg_h
        h_xg = h_att * a_def * avg_h
        a_xg = a_att * h_def * avg_a
        xg_stats = [h_xg, a_xg, h_xg - a_xg]

    # 3. H2H
    h2h_m = history[((history['HomeTeam']==home) & (history['AwayTeam']==away)) | 
                    ((history['HomeTeam']==away) & (history['AwayTeam']==home))].tail(6)
    pts = 0
    if len(h2h_m) > 0:
        for _, m in h2h_m.iterrows():
            if m['HomeTeam'] == home: pts += 3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0)
            elif m['AwayTeam'] == home: pts += 3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0)
        h2h_score = pts/len(h2h_m)
    else: h2h_score = 1.0

    # 4. Strict Form & Overall Form
    sp_h = history[history['HomeTeam'] == home].tail(6)
    h_form = sum([3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0) for _, m in sp_h.iterrows()])
    
    sp_a = history[history['AwayTeam'] == away].tail(6)
    a_form = sum([3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0) for _, m in sp_a.iterrows()])

    # รวม Features ให้ตรงลำดับตอนเทรน
    features = [oh, od, oa] + xg_stats + [h2h_score, h_form, a_form, imp_h, imp_a]
    cols = ['AvgH', 'AvgD', 'AvgA', 'Home_xG', 'Away_xG', 'xG_Diff', 'H2H', 'Home_Form', 'Away_Form', 'Imp_H', 'Imp_A']
    
    return pd.DataFrame([features], columns=cols)

# =========================================================
# 3. ✍️ โซนกรอกคู่บอล (INPUT ZONE)
# =========================================================
# กรอกชื่อทีมให้ตรงกับในเว็บ football-data.co.uk
upcoming_matches = [
    {"Home": "Liverpool",       "Away": "Barnsley",  "Odds": [1.102,8.9,19]}, 

]

# =========================================================
# 4. ประมวลผลและแสดงผล
# =========================================================
print(f"\n{'='*100}")
print(f"{'MATCH':<30} | {'HOME':<6} | {'DRAW':<6} | {'AWAY':<6} | {'PREDICTION':<15} | {'GRADE'}")
print(f"{'='*100}")

for match in upcoming_matches:
    home, away, odds = match['Home'], match['Away'], match['Odds']
    
    # Check Team Exist
    if home not in full_data['HomeTeam'].unique() or away not in full_data['AwayTeam'].unique():
        print(f"{home} vs {away} : ❌ ไม่พบชื่อทีม (เช็คตัวสะกด)")
        continue

    # Prepare Data
    input_df = prepare_match_features(home, away, odds[0], odds[1], odds[2], full_data)
    
    # Predict
    proba = model.predict_proba(input_df)[0]
    prob_h, prob_d, prob_a = proba[2]*100, proba[1]*100, proba[0]*100
    
    # Logic การทาย
    max_prob = max(prob_h, prob_d, prob_a)
    if max_prob == prob_h: pred = f"Home ({home})"
    elif max_prob == prob_a: pred = f"Away ({away})"
    else: pred = "Draw"

    # ตัดเกรดความมั่นใจ (อิงจากตาราง Evaluation ที่เราทำ)
    grade = ""
    if max_prob >= 75: grade = "🔥 GOD"
    elif max_prob >= 65: grade = "💎 SOLID"
    elif max_prob >= 55: grade = "✅ GOOD"
    elif max_prob >= 50: grade = "🤔 FAIR"
    else: grade = "😐 RISKY"
    
    # สี Highlight (ถ้าเป็น God หรือ Solid)
    if "GOD" in grade or "SOLID" in grade:
        match_str = f"👉 {home} vs {away}"
    else:
        match_str = f"{home} vs {away}"

    print(f"{match_str:<30} | {prob_h:.1f}%  | {prob_d:.1f}%  | {prob_a:.1f}%  | {pred:<15} | {grade} ({max_prob:.1f}%)")

print(f"{'='*100}")