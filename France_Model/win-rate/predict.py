import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO

# =========================================================
# 1. โหลดฐานข้อมูล & โมเดล
# =========================================================
print("📂 กำลังโหลดฐานข้อมูลประวัติย้อนหลัง...")
urls = [ "https://www.football-data.co.uk/mmz4281/2526/F1.csv", # Future/Current
    "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
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

try:
    model = joblib.load('model_v5_france.pkl')
    print("🧠 โหลดสมอง V5.0 (Strict + Momentum) เรียบร้อย!")
except:
    print("❌ ไม่พบโมเดล! กรุณารัน Part 1 เพื่อสร้างสมอง V5 ก่อน"); exit()

# =========================================================
# 2. ฟังก์ชันคำนวณพลัง (อัปเกรด V5)
# =========================================================
def calculate_features(home, away, oh, od, oa, history):
    # 1. Market Wisdom
    imp_h, imp_a = (1/oh)*100, (1/oa)*100

    # 2. xG Calculation
    avg_h, avg_a = history['FTHG'].mean(), history['FTAG'].mean()
    h_games = history[history['HomeTeam'] == home].tail(10)
    a_games = history[history['AwayTeam'] == away].tail(10)

    if len(h_games) < 3: xg_stats = [1.3, 1.3, 0]
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
    for _, m in h2h_m.iterrows():
        if m['HomeTeam'] == home: pts += 3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0)
        elif m['AwayTeam'] == home: pts += 3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0)
    h2h_score = pts/len(h2h_m) if len(h2h_m) > 0 else 1.0

    # 4. Strict Form
    sp_h = history[history['HomeTeam'] == home].tail(6)
    h_strict = sum([3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0) for _, m in sp_h.iterrows()])
    
    sp_a = history[history['AwayTeam'] == away].tail(6)
    a_strict = sum([3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0) for _, m in sp_a.iterrows()])

    # 5. 🔥 NEW: Overall Form (Momentum)
    # ---------------------------------------------------------
    last5_h = history[(history['HomeTeam'] == home) | (history['AwayTeam'] == home)].tail(5)
    h_overall = 0
    for _, m in last5_h.iterrows():
        if m['HomeTeam'] == home: h_overall += 3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0)
        else: h_overall += 3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0)

    last5_a = history[(history['HomeTeam'] == away) | (history['AwayTeam'] == away)].tail(5)
    a_overall = 0
    for _, m in last5_a.iterrows():
        if m['HomeTeam'] == away: a_overall += 3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0)
        else: a_overall += 3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0)
    # ---------------------------------------------------------

    features = [oh, od, oa] + xg_stats + [h2h_score, h_strict, a_strict, h_overall, a_overall, imp_h, imp_a]
    cols = ['AvgH', 'AvgD', 'AvgA', 'Home_xG', 'Away_xG', 'xG_Diff', 'H2H', 'Home_Strict', 'Away_Strict', 'Home_Overall', 'Away_Overall', 'Imp_H', 'Imp_A']
    
    return pd.DataFrame([features], columns=cols)

# =========================================================
# 3. ✍️ โซนกรอกคู่บอล (Update ราคาและคู่ได้เลย)
# =========================================================
upcoming_matches = [ 
   {"Home": "Lyon",           "Away": "Lille",        "Odds": [2.125, 3.66, 3.68]},
    {"Home": "Toulouse",       "Away": "Auxerre",      "Odds": [1.818, 3.575, 5.38]},
    {"Home": "Nice",           "Away": "Brest",        "Odds": [2.308, 3.475, 3.395]},
    {"Home": "Angers",         "Away": "Metz",         "Odds": [2.296, 3.35, 3.555]},
    {"Home": "Strasbourg",     "Away": "Paris SG",          "Odds": [5.13, 4.525, 1.668]},

   
]

# =========================================================
# 4. ประมวลผล
# =========================================================
print(f"\n{'='*95}")
print(f"{'MATCH':<30} | {'HOME %':<8} | {'DRAW %':<8} | {'AWAY %':<8} | {'PREDICTION':<12}")
print(f"{'='*95}")

for match in upcoming_matches:
    home, away, odds = match['Home'], match['Away'], match['Odds']
    
    # เช็คว่ามีทีมนี้ใน Database ไหม
    if home not in full_data['HomeTeam'].unique() or away not in full_data['AwayTeam'].unique():
        print(f"{home} vs {away} : ❌ ไม่พบชื่อทีม (ตรวจสอบตัวสะกด)")
        continue

    input_df = calculate_features(home, away, odds[0], odds[1], odds[2], full_data)
    
    proba = model.predict_proba(input_df)[0]
    prob_a, prob_d, prob_h = proba[0]*100, proba[1]*100, proba[2]*100
    
    max_prob = max(prob_a, prob_d, prob_h)
    if max_prob == prob_h: pred = "Home 🏠"
    elif max_prob == prob_a: pred = "Away ✈️"
    else: pred = "Draw 🤝"

    marker = "🔥 SUPER" if max_prob >= 75 else ("✅ PLAY" if max_prob >= 60 else "")
    
    print(f"{home} vs {away:<15} | {prob_h:.1f}%   | {prob_d:.1f}%   | {prob_a:.1f}%   | {pred:<10} {marker}")

print(f"{'='*95}")