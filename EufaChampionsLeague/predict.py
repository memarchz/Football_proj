import pandas as pd
import numpy as np
import joblib

# =========================================================
# 1. โหลดระบบ
# =========================================================
print("📂 กำลังโหลดฐานข้อมูลยุโรป (Europe DB)...")
try:
    full_data = pd.read_pickle('europe_db.pkl')
    model = joblib.load('model_uefa_sniper.pkl')
    print("🧠 โหลดสมอง V5.0 EUROPE เรียบร้อย!")
except:
    print("❌ ไม่พบไฟล์! กรุณารัน Part 1 ก่อน"); exit()

# =========================================================
# 2. ฟังก์ชันคำนวณ (เหมือนเดิมแต่ใช้ DB ยุโรป)
# =========================================================
def calculate_features(home, away, oh, od, oa, history):
    # ใช้ Logic เดียวกับตอนเทรนเป๊ะๆ
    imp_h, imp_a = (1/oh)*100, (1/oa)*100
    
    # ดึงค่าเฉลี่ยรวมของยุโรป
    avg_h_league = history['FTHG'].mean()
    avg_a_league = history['FTAG'].mean()

    h_games = history[history['HomeTeam'] == home].tail(10)
    a_games = history[history['AwayTeam'] == away].tail(10)

    if len(h_games) < 3 or len(a_games) < 3: 
        print(f"⚠️ ข้อมูลทีม {home} หรือ {away} น้อยเกินไป (อาจจะเป็นทีมนอก 5 ลีกใหญ่)")
        xg_stats = [1.3, 1.3, 0] # ค่า Default
    else:
        h_att = h_games['FTHG'].mean() / avg_h_league
        h_def = h_games['FTAG'].mean() / avg_a_league
        a_att = a_games['FTAG'].mean() / avg_a_league
        a_def = a_games['FTHG'].mean() / avg_h_league
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

    # Overall Form (Momentum)
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
# 3. ✍️ โซนกรอกคู่บอล UEFA (ตัวอย่าง)
# =========================================================
# ใส่ชื่อทีมภาษาอังกฤษให้ตรง (ระวัง: Inter, AC Milan, Bayern Munich, Real Madrid)
uefa_matches = [
    {"Home": "Arsenal",       "Away": "Girona",         "Odds": [1.067, 21.0, 32.0]}, # ราคาโหดมาก
    {"Home": "Monaco",        "Away": "Juventus",       "Odds": [4.525, 3.845, 1.872]},
    {"Home": "Ein Frankfurt", "Away": "Tottenham",      "Odds": [4.775, 4.215, 1.758]},
    {"Home": "St Gilloise",   "Away": "Atalanta",       "Odds": [3.66, 3.81, 2.084]}, # Union SG vs Atalanta
    {"Home": "PSV Eindhoven", "Away": "Bayern Munich",  "Odds": [5.64, 4.825, 1.585]},
    {"Home": "Man City",      "Away": "Galatasaray",    "Odds": [1.281, 7.6, 9.65]},
    {"Home": "Dortmund",      "Away": "Inter",          "Odds": [3.11, 3.635, 2.388]}, # Inter Milan
    {"Home": "Leverkusen",    "Away": "Villarreal",     "Odds": [1.787, 4.265, 4.515]},
    {"Home": "Benfica",       "Away": "Real Madrid",    "Odds": [4.215, 4.265, 1.839]},
    {"Home": "Club Brugge",   "Away": "Marseille",      "Odds": [2.613, 3.84, 2.685]},
    {"Home": "Napoli",        "Away": "Chelsea",        "Odds": [3.74, 3.7, 2.094]},
    {"Home": "Barcelona",     "Away": "FC Copenhagen",  "Odds": [1.156, 10.7, 17.5]},
    {"Home": "Paris SG",      "Away": "Newcastle",      "Odds": [1.666, 4.975, 4.67]}, # PSG
    {"Home": "Liverpool",     "Away": "Qarabag",        "Odds": [1.141, 12.0, 17.5]}, # Qarabag อาจไม่มีใน DB
    {"Home": "Ath Madrid",    "Away": "Bodo Glimt",     "Odds": [1.301, 7.21, 9.25]}, # Atletico
    {"Home": "Ath Bilbao",    "Away": "Sp Lisbon",      "Odds": [2.855, 3.475, 2.652]}, # Sporting CP
    {"Home": "Ajax",          "Away": "Olympiakos",     "Odds": [3.29, 3.615, 2.297]}
]

# =========================================================
# 4. แสดงผลทำนาย
# =========================================================
print(f"\n{'='*95}")
print(f"{'UEFA MATCH':<30} | {'HOME %':<8} | {'DRAW %':<8} | {'AWAY %':<8} | {'PREDICTION':<12}")
print(f"{'='*95}")

for match in uefa_matches:
    home, away, odds = match['Home'], match['Away'], match['Odds']
    
    # เช็คว่ามีทีมนี้ใน Database ไหม (Big 5 Leagues)
    if home not in full_data['HomeTeam'].unique():
        print(f"{home:<30} : ❌ ไม่พบข้อมูล (อาจไม่ใช่ทีมจาก 5 ลีกใหญ่ หรือสะกดผิด)")
        continue
    if away not in full_data['AwayTeam'].unique():
        print(f"{away:<30} : ❌ ไม่พบข้อมูล (อาจไม่ใช่ทีมจาก 5 ลีกใหญ่ หรือสะกดผิด)")
        continue

    input_df = calculate_features(home, away, odds[0], odds[1], odds[2], full_data)
    
    proba = model.predict_proba(input_df)[0]
    prob_a, prob_d, prob_h = proba[0]*100, proba[1]*100, proba[2]*100
    
    max_prob = max(prob_a, prob_d, prob_h)
    if max_prob == prob_h: pred = "Home 🏠"
    elif max_prob == prob_a: pred = "Away ✈️"
    else: pred = "Draw 🤝"

    marker = "🔥 SUPER" if max_prob >= 70 else ("✅ PLAY" if max_prob >= 55 else "")
    
    print(f"{home} vs {away:<15} | {prob_h:.1f}%   | {prob_d:.1f}%   | {prob_a:.1f}%   | {pred:<10} {marker}")

print(f"{'='*95}")