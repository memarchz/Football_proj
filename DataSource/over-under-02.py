import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import poisson

# =========================================================
# 🏛️ SYSTEM V7.0: THE MARKET CONSENSUS
# =========================================================
print("🚀 Initiating V7.0: 4-Factor Oracle (Elo + Poisson + Form + Market)...")

# 1. โหลดข้อมูล (เหมือนเดิม)
urls = [
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv"
]
dfs = []
for url in urls:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            d = pd.read_csv(StringIO(r.content.decode('latin-1')))
            dfs.append(d)
    except: pass

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).sort_values(by='Date').reset_index(drop=True)

# =========================================================
# 🧠 CORE ENGINES (Elo & Poisson & Form)
# =========================================================
# --- 1. Elo Engine ---
elo_ratings = {}
def get_elo(team): return elo_ratings.get(team, 1500)
def update_elo(home, away, goal_h, goal_a):
    k = 30
    r_h, r_a = get_elo(home), get_elo(away)
    exp_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
    act_h = 1 if goal_h > goal_a else 0.5 if goal_h == goal_a else 0
    elo_ratings[home] = r_h + k * (act_h - exp_h)
    elo_ratings[away] = r_a + k * ((1 - act_h) - (1 - exp_h))

for idx, row in df.iterrows():
    update_elo(row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'])

# --- 2. Poisson Engine ---
avg_h = df['FTHG'].mean()
avg_a = df['FTAG'].mean()
def get_poisson_stats(home, away):
    h_m = df[df['HomeTeam'] == home].tail(10)
    a_m = df[df['AwayTeam'] == away].tail(10)
    if len(h_m) < 5 or len(a_m) < 5: return None
    
    h_att = h_m['FTHG'].mean() / avg_h
    h_def = h_m['FTAG'].mean() / avg_a
    a_att = a_m['FTAG'].mean() / avg_a
    a_def = a_m['FTHG'].mean() / avg_h
    
    xg_h = h_att * a_def * avg_h
    xg_a = a_att * h_def * avg_a
    return xg_h, xg_a

# --- 3. Form Engine ---
def get_form(team):
    matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
    pts = 0
    for _, m in matches.iterrows():
        if m['HomeTeam'] == team: pts += 3 if m['FTR'] == 'H' else 1 if m['FTR'] == 'D' else 0
        else: pts += 3 if m['FTR'] == 'A' else 1 if m['FTR'] == 'D' else 0
    return pts

# =========================================================
# ✍️ INPUT ZONE: ใส่ราคาบ่อนที่นี่ (สำคัญมาก!)
# =========================================================
# ตัวอย่าง: ใส่ราคาจากเว็บพนันจริง (Odds)
upcoming_matches = [
    {
        "Home": "Man United", "Away": "Liverpool", 
        "Odds_H": 3.50, "Odds_A": 2.00,  # ราคา Win/Loss
        "Odds_O25": 1.50, "Odds_U25": 2.60, # ราคา Over/Under 2.5
        "Odds_O35": 2.30, "Odds_U35": 1.60  # ราคา Over/Under 3.5
    },
    {
        "Home": "Arsenal", "Away": "Wolves", 
        "Odds_H": 1.25, "Odds_A": 10.0,
        "Odds_O25": 1.65, "Odds_U25": 2.20,
        "Odds_O35": 2.60, "Odds_U35": 1.45
    },
    {
        "Home": "Everton", "Away": "Nott'm Forest", 
        "Odds_H": 2.10, "Odds_A": 3.40,
        "Odds_O25": 1.95, "Odds_U25": 1.85, # ราคาใกล้กัน บ่อนมองสูสี
        "Odds_O35": 3.40, "Odds_U35": 1.30  # ราคาต่ำ 3.5 จ่ายน้อยมาก (บ่อนมั่นใจว่ายิงไม่ถึง 4)
    },
    {
        "Home": "Brentford", "Away": "Aston Villa", 
        "Odds_H": 2.90, "Odds_A": 2.30,
        "Odds_O25": 1.55, "Odds_U25": 2.45, 
        "Odds_O35": 2.40, "Odds_U35": 1.55 
    }
]

# =========================================================
# ⚖️ V7.0 DECISION LOGIC (4 Factors)
# =========================================================
print(f"\n{'='*115}")
print(f"{'MATCH':<25} | {'xG Total':<8} | {'MKT VIEW':<15} | {'THE ORACLE VERDICT (Final Decision)'}")
print(f"{'-'*115}")

for m in upcoming_matches:
    h, a = m['Home'], m['Away']
    
    # 1. Calculate Stats
    xg_res = get_poisson_stats(h, a)
    if not xg_res: continue
    xg_h, xg_a = xg_res
    total_xg = xg_h + xg_a
    
    elo_h, elo_a = get_elo(h), get_elo(a)
    form_h, form_a = get_form(h), get_form(a)
    
    # 2. Market Analysis (มุมมองเจ้ามือ)
    # แปลง Odds เป็น Implied Probability (%)
    # ถ้า Odds < 1.80 แปลว่าเจ้ามือมองว่ามีโอกาสเกิดสูง (>55%)
    market_view = []
    if m['Odds_O25'] < 1.70: market_view.append("Expect Goals")
    elif m['Odds_U25'] < 1.70: market_view.append("Expect Tight")
    
    if m['Odds_H'] < 1.50: market_view.append("Home Fav")
    elif m['Odds_A'] < 1.50: market_view.append("Away Fav")
    
    mkt_str = ", ".join(market_view) if market_view else "Neutral"

    # 3. 🎯 FINAL DECISION LOGIC (Consensus of 4)
    decision = "⛔ PASS"
    confidence = ""

    # --- WINNER PREDICTION ---
    # Home Win: Elo นำ + ฟอร์มดี + xG ชนะขาด + ราคาจ่ายต่ำ(ยืนยันว่าเก่งจริง)
    if (elo_h > elo_a + 50) and (form_h > form_a) and (xg_h > xg_a + 0.5) and (m['Odds_H'] < 2.10):
        decision = f"🏆 {h} WIN"
    
    elif (elo_a > elo_h + 50) and (form_a > form_h) and (xg_a > xg_h + 0.5) and (m['Odds_A'] < 2.10):
        decision = f"🏆 {a} WIN"

    # --- GOAL PREDICTION (Over/Under) ---
    # เงื่อนไข OVER 2.5:
    # 1. Poisson: xG รวมต้อง > 2.8 (ยิงกันยับ)
    # 2. Market: ราคา Over 2.5 ต้องต่ำกว่า 1.85 (บ่อนกลัว Over)
    # 3. Form: ทั้งคู่ฟอร์มยิงประตูใช้ได้ (ไม่ใช่ 0-0 มาตลอด)
    if (total_xg >= 2.80) and (m['Odds_O25'] < 1.85):
        decision = "🔥 OVER 2.5 (High Conf)"
        
    # เงื่อนไข UNDER 2.5:
    # 1. Poisson: xG รวมต้อง < 2.30 (ฝืดจัด)
    # 2. Market: ราคา Under 2.5 ต้องต่ำกว่า 1.90
    elif (total_xg <= 2.30) and (m['Odds_U25'] < 1.90):
        decision = "🧊 UNDER 2.5 (Tight Game)"
        
    # เงื่อนไข UNDER 3.5 (Safety Net - ทีเด็ดบอลรองสกอร์):
    # 1. ถ้า xG ไม่สูงมาก (< 3.0)
    # 2. และราคา Under 3.5 ต่ำมาก (< 1.45) แสดงว่าบ่อนมั่นใจว่าไม่เละ
    elif (total_xg < 3.0) and (m['Odds_U35'] < 1.50) and decision == "⛔ PASS":
        decision = "🛡️ UNDER 3.5 (Safe Bet)"

    # Print Result
    print(f"{h:<12} vs {a:<9} | {total_xg:.2f}     | {mkt_str:<15} | {decision}")

print(f"{'='*115}")
print("📌 หลักการทำงาน V7.0:")
print("1. 🔥 OVER 2.5: เกิดเมื่อสถิติชี้ว่ายิงแน่ (xG>2.8) และราคาบ่อนจ่ายน้อย (Odds < 1.85)")
print("2. 🧊 UNDER 2.5: เกิดเมื่อสถิติชี้ว่าฝืด (xG<2.3) และราคาบ่อนสนับสนุน")
print("3. 🛡️ UNDER 3.5: เป็นตัวเลือก Play Safe เมื่อบอลดูไม่น่าขาด แต่ไม่กล้ากดต่ำ 2.5")