import pandas as pd
import numpy as np
import joblib

# =========================================================
# 1. โหลดระบบ (ต้องรัน Part 1 เพื่อสร้างไฟล์ก่อน)
# =========================================================
print("📂 กำลังโหลดฐานข้อมูล Europa League...")
try:
    full_data = pd.read_pickle('europe_db.pkl')
    # โหลดโมเดล Europa ที่เทรนไว้
    model = joblib.load('model_europa_sniper.pkl') 
    print("🧠 โหลดสมอง V5.5 EUROPA EDITION เรียบร้อย!")
except:
    print("❌ ไม่พบไฟล์! กรุณารัน Part 1 (Training) ให้เสร็จก่อนครับ"); exit()

# =========================================================
# 2. ฟังก์ชันคำนวณพลัง (ต้องตรงกับตอน Train เป๊ะๆ)
# =========================================================
def calculate_europa_features(home, away, oh, od, oa, history):
    # Market Impulse
    imp_h, imp_a = (1/oh)*100, (1/oa)*100
    
    # ใช้ข้อมูลทั้งหมดที่มีใน DB เป็น "อดีต"
    avg_h_league = history['FTHG'].mean()
    avg_a_league = history['FTAG'].mean()

    h_games = history[history['HomeTeam'] == home].tail(10)
    a_games = history[history['AwayTeam'] == away].tail(10)

    # Logic สำหรับทีมเล็ก/ทีมที่ข้อมูลน้อย (Europa Logic)
    if len(h_games) < 3 or len(a_games) < 3: 
        # ถ้าข้อมูลน้อย ให้ใช้ค่ากลางๆ หรือวิเคราะห์จากราคาแทน
        xg_stats = [0, 0, 0] 
    else:
        h_att = h_games['FTHG'].mean() / avg_h_league if avg_h_league > 0 else 1
        h_def = h_games['FTAG'].mean() / avg_a_league if avg_a_league > 0 else 1
        a_att = a_games['FTAG'].mean() / avg_a_league if avg_a_league > 0 else 1
        a_def = a_games['FTHG'].mean() / avg_h_league if avg_h_league > 0 else 1
        h_xg = h_att * a_def * avg_h_league
        a_xg = a_att * h_def * avg_a_league
        xg_stats = [h_xg, a_xg, h_xg - a_xg]

    # H2H
    h2h_m = history[((history['HomeTeam']==home) & (history['AwayTeam']==away)) | 
                    ((history['HomeTeam']==away) & (history['AwayTeam']==home))].tail(6)
    pts = 0
    if len(h2h_m) > 0:
        for _, m in h2h_m.iterrows():
            if m['HomeTeam'] == home: pts += 3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0)
            elif m['AwayTeam'] == home: pts += 3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0)
        h2h_score = pts/len(h2h_m)
    else: h2h_score = 1.0

    # Strict Form
    sp_h = history[history['HomeTeam'] == home].tail(6)
    h_strict = sum([3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0) for _, m in sp_h.iterrows()])
    sp_a = history[history['AwayTeam'] == away].tail(6)
    a_strict = sum([3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0) for _, m in sp_a.iterrows()])

    # Overall Form
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

    features = [oh, od, oa] + xg_stats + [h2h_score, h_strict, a_strict, h_overall, a_overall, imp_h, imp_a]
    cols = ['AvgH', 'AvgD', 'AvgA', 'Home_xG', 'Away_xG', 'xG_Diff', 'H2H', 'Home_Strict', 'Away_Strict', 'Home_Overall', 'Away_Overall', 'Imp_H', 'Imp_A']
    
    return pd.DataFrame([features], columns=cols)

# =========================================================
# 3. ✍️ โซนกรอกคู่บอล (จากรูปภาพ 3 & 4)
# =========================================================
# หมายเหตุ: ทีมบางทีม (เช่น Viktoria Plzen, Maccabi) อาจไม่มีใน DB หากไม่ได้โหลดลีกนั้นมา
uefa_matches = [
    # --- จากรูปที่ 3 ---
    {"Home": "Crvena Zvezda",  "Away": "Celta Vigo",   "Odds": [2.343, 3.745, 3.11]},
    {"Home": "FCSB",           "Away": "Fenerbahce",   "Odds": [5.98, 4.6, 1.585]},
    {"Home": "Go Ahead Eagles","Away": "Braga",        "Odds": [5.05, 4.19, 1.727]},
    {"Home": "Maccabi Tel Aviv","Away": "Bologna",     "Odds": [8.9, 5.27, 1.402]},
    {"Home": "Sturm Graz",     "Away": "SK Brann",     "Odds": [2.941, 3.515, 2.557]},
    {"Home": "Genk",           "Away": "Malmo FF",     "Odds": [1.339, 5.77, 10.4]},
    {"Home": "Celtic",         "Away": "Utrecht",      "Odds": [1.401, 5.31, 8.75]},
    {"Home": "Stuttgart",      "Away": "Young Boys",   "Odds": [1.214, 7.95, 14.9]},
    {"Home": "Betis",          "Away": "Feyenoord",    "Odds": [1.698, 4.45, 4.96]},
    
    # --- Europa League (More Matches 30/01) ---
    {"Home": "Aston Villa",    "Away": "Salzburg",     "Odds": [1.469, 5.23, 6.96]},
    {"Home": "Porto",          "Away": "Rangers",      "Odds": [1.264, 6.81, 12.9]},
    {"Home": "Nottm Forest",   "Away": "Ferencvaros",  "Odds": [1.586, 4.245, 6.7]},
    {"Home": "Basel",          "Away": "Viktoria Plzen","Odds": [2.115, 3.885, 3.505]},
    {"Home": "Panathinaikos",  "Away": "Roma",         "Odds": [5.55, 3.845, 1.738]},
    {"Home": "Midtjylland",    "Away": "Dinamo Zagreb","Odds": [1.921, 3.8, 4.315]},
    {"Home": "LASK",           "Away": "Freiburg",     "Odds": [2.043, 3.635, 3.985]},
    {"Home": "Ludogorets",     "Away": "Nice",         "Odds": [1.851, 4.005, 4.43]},
    {"Home": "Lyon",           "Away": "PAOK",         "Odds": [2.299, 3.54, 3.355]}
]

# =========================================================
# 4. แสดงผลทำนาย (Europa Edition)
# =========================================================
print(f"\n{'='*100}")
print(f"{'EUROPA MATCH':<30} | {'HOME':<6} | {'DRAW':<6} | {'AWAY':<6} | {'PREDICTION':<15}")
print(f"{'='*100}")

for match in uefa_matches:
    home, away, odds = match['Home'], match['Away'], match['Odds']
    
    # ตรวจสอบว่ามีข้อมูลทีมใน Database หรือไม่
    # หมายเหตุ: เรายอมให้ขาด 1 ทีมได้ (เช่น Freiburg เจอ ทีมเล็ก) โดยจะใช้ stats ของทีมที่มี + ราคาเป็นหลัก
    h_exist = home in full_data['HomeTeam'].unique()
    a_exist = away in full_data['AwayTeam'].unique()
    
    if not h_exist and not a_exist:
        print(f"{home:<12} vs {away:<12} : ⚠️ ข้าม (ไม่พบข้อมูลทั้ง 2 ทีม)")
        continue

    # คำนวณ Features
    input_df = calculate_europa_features(home, away, odds[0], odds[1], odds[2], full_data)
    
    # Predict
    proba = model.predict_proba(input_df)[0]
    prob_a, prob_d, prob_h = proba[0]*100, proba[1]*100, proba[2]*100
    
    max_prob = max(prob_a, prob_d, prob_h)
    
    if max_prob == prob_h: pred = "Home Win 🏠"
    elif max_prob == prob_a: pred = "Away Win ✈️"
    else: pred = "Draw 🤝"

    # ใส่เกรดความมั่นใจ
    marker = ""
    if max_prob >= 72: marker = "🔥 SUPER"
    elif max_prob >= 60: marker = "✅ PLAY"
    elif max_prob < 45: marker = "💀 SKIP" # บอลสูสีเกินไป

    print(f"{home:<12} vs {away:<12} | {prob_h:.0f}%   | {prob_d:.0f}%   | {prob_a:.0f}%   | {pred:<10} {marker}")

print(f"{'='*100}")
print("📝 Note: สำหรับคู่ที่เจอทีมเล็ก (ไม่มีใน DB) โมเดลจะใช้ 'ราคาต่อรอง' เป็นตัวนำทางหลักครับ")