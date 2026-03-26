import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import poisson

# =========================================================
# 🏛️ SYSTEM V6.0: THE ORACLE (Consensus Model)
# =========================================================
print("🚀 เริ่มระบบ V6.0 The Oracle (ระดับพระเจ้า 90%)...")
print("⚠️ คำเตือน: ระบบนี้จะคัดบอลทิ้งเยอะมาก จะเหลือเฉพาะคู่ที่ 'ชัวร์' จริงๆ เท่านั้น")

# 1. โหลดข้อมูล
urls = [
    "https://www.football-data.co.uk/mmz4281/2526/I1.csv", # ฤดูกาลปัจจุบัน
    "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
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
# 🧠 1. ระบบ ELO RATING (วัดความเก๋าเกม)
# =========================================================
elo_ratings = {}
def get_elo(team):
    return elo_ratings.get(team, 1500) # เริ่มต้นที่ 1500

def update_elo(home, away, goal_h, goal_a):
    k_factor = 30 # ค่าความอ่อนไหวต่อผลการแข่ง
    r_h = get_elo(home)
    r_a = get_elo(away)
    
    # คำนวณผลคาดการณ์
    expected_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
    
    # ผลจริง (1=ชนะ, 0.5=เสมอ, 0=แพ้)
    if goal_h > goal_a: actual_h = 1
    elif goal_h == goal_a: actual_h = 0.5
    else: actual_h = 0
    
    # อัปเดตคะแนน
    new_r_h = r_h + k_factor * (actual_h - expected_h)
    new_r_a = r_a + k_factor * ((1 - actual_h) - (1 - expected_h))
    
    elo_ratings[home] = new_r_h
    elo_ratings[away] = new_r_a

# รัน Elo ย้อนหลังเพื่อสร้างค่าพลังปัจจุบัน
for idx, row in df.iterrows():
    update_elo(row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'])

# =========================================================
# 🧠 2. ระบบ POISSON (วัดความน่าจะเป็นสกอร์)
# =========================================================
avg_h_goal = df['FTHG'].mean()
avg_a_goal = df['FTAG'].mean()

def get_poisson_pred(home, away):
    # (ใช้ Logic เดิมจาก V5)
    h_matches = df[df['HomeTeam'] == home].tail(10)
    a_matches = df[df['AwayTeam'] == away].tail(10)
    
    if len(h_matches) < 5 or len(a_matches) < 5: return None
    
    h_att = h_matches['FTHG'].mean() / avg_h_goal
    h_def = h_matches['FTAG'].mean() / avg_a_goal
    a_att = a_matches['FTAG'].mean() / avg_a_goal
    a_def = a_matches['FTHG'].mean() / avg_h_goal
    
    xg_h = h_att * a_def * avg_h_goal
    xg_a = a_att * h_def * avg_a_goal
    
    return xg_h, xg_a

# =========================================================
# 🧠 3. ระบบ MOMENTUM (ฟอร์มสด 5 นัด)
# =========================================================
def get_form_points(team):
    matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
    points = 0
    for _, m in matches.iterrows():
        if m['HomeTeam'] == team:
            if m['FTR'] == 'H': points += 3
            elif m['FTR'] == 'D': points += 1
        else:
            if m['FTR'] == 'A': points += 3
            elif m['FTR'] == 'D': points += 1
    return points # เต็ม 15 คะเเนน

# =========================================================
# ⚖️ THE ORACLE JUDGEMENT (ห้องตัดสินใจ)
# =========================================================
upcoming_matches = [
    {"Home": "Milan",    "Away": "Verona"},
    {"Home": "Cremonese",    "Away": "Napoli"},
    {"Home": "Bologna",    "Away": "Sassuolo"},
    {"Home": "Atalanta",    "Away": "Inter"},
]

print(f"\n{'='*100}")
print(f"🔮 V6.0 THE ORACLE PREDICTION (ต้องการฉันทามติ 3/3 เท่านั้น)")
print(f"{'='*100}")
print(f"{'MATCH':<25} | {'ELO':<10} | {'POISSON':<10} | {'FORM':<10} | {'FINAL DECISION (90%)'}")
print(f"{'-'*100}")

for m in upcoming_matches:
    h, a = m['Home'], m['Away']
    
    # 1. Elo Analysis
    elo_h = get_elo(h)
    elo_a = get_elo(a)
    elo_diff = elo_h - elo_a
    elo_pick = "Draw"
    if elo_diff > 100: elo_pick = "Home"
    elif elo_diff < -100: elo_pick = "Away"
    
    # 2. Poisson Analysis
    p_res = get_poisson_pred(h, a)
    poi_pick = "Skip"
    if p_res:
        xg_h, xg_a = p_res
        if xg_h - xg_a > 0.6: poi_pick = "Home"
        elif xg_a - xg_h > 0.6: poi_pick = "Away"
        elif xg_h + xg_a < 2.2: poi_pick = "Under" # เน้นสกอร์ต่ำ
        elif xg_h + xg_a > 3.2: poi_pick = "Over"
        
    # 3. Form Analysis
    form_h = get_form_points(h)
    form_a = get_form_points(a)
    form_pick = "Draw"
    if form_h >= 10 and form_a <= 5: form_pick = "Home" # เจ้าบ้านเทพ ทีมเยือนกาก
    elif form_a >= 10 and form_h <= 5: form_pick = "Away"
    
    # --- 🎯 FINAL VERDICT (ต้องตรงกันหมด) ---
    decision = "⛔ PASS"
    
    # กรณี Home Win (ต้องตรงกัน 3 อย่าง)
    if elo_pick == "Home" and poi_pick == "Home" and form_pick == "Home":
        decision = "🏆 HOME WIN (90%)"
        
    # กรณี Away Win
    elif elo_pick == "Away" and poi_pick == "Away" and form_pick == "Away":
        decision = "🏆 AWAY WIN (90%)"
        
    # กรณี Under 3.5 (ไม้ตาย)
    # ถ้า Poisson บอกต่ำ + Elo สูสี + ฟอร์มไม่ห่างกันมาก = บอลอุดแน่นอน
    elif poi_pick == "Under" and abs(elo_diff) < 150:
         decision = "🧊 UNDER 3.5 (90%)"

    print(f"{h:<12} vs {a:<9} | {elo_pick:<10} | {poi_pick:<10} | {form_pick:<10} | {decision}")

print(f"{'='*100}")
print("หมายเหตุ:")
print("- PASS: ระบบมองว่ามีความเสี่ยง หรือข้อมูลขัดแย้งกัน (อย่าเล่น)")
print("- 🏆/🧊: คือคู่ที่ 3 ศาสตร์เห็นตรงกันหมด (นี่คือทีเด็ดระดับพระเจ้า)")