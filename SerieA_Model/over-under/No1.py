import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings

# ปิด Warning เพื่อความสะอาดของ Output
warnings.filterwarnings('ignore')

# =========================================================
# 1. LOAD DATA & SETUP
# =========================================================
print("🔄 V17.5: Loading Data & Calibrating System...")

urls = [
    # Serie A
    "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2223/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2122/I1.csv"
]

dfs = []
for url in urls:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            csv_data = StringIO(r.content.decode('latin-1'))
            df = pd.read_csv(csv_data)
            df.columns = df.columns.str.strip()
            # เลือกเฉพาะคอลัมน์ที่ต้องใช้
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            if all(c in df.columns for c in cols):
                dfs.append(df[cols])
    except: pass

if not dfs:
    print("❌ Error: โหลดข้อมูลไม่ได้ (เช็คเน็ต)"); exit()

full_df = pd.concat(dfs, ignore_index=True)
full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')
full_df = full_df.dropna(subset=['Date', 'FTHG', 'FTAG']).sort_values('Date').reset_index(drop=True)

print(f"✅ Loaded {len(full_df)} matches.")

# =========================================================
# 2. 🛠️ CORE LOGIC: SCORING SYSTEM
# =========================================================

def calculate_stats(df_hist, team_name):
    if len(df_hist) < 3: return None
    
    scored = []
    conceded = []
    total_goals_list = []
    btts_count = 0
    over25_count = 0
    
    for _, m in df_hist.iterrows():
        if m['HomeTeam'] == team_name:
            g_for, g_ag = m['FTHG'], m['FTAG']
        else:
            g_for, g_ag = m['FTAG'], m['FTHG']
            
        scored.append(g_for)
        conceded.append(g_ag)
        total = g_for + g_ag
        total_goals_list.append(total)
        
        if (g_for > 0) and (g_ag > 0): btts_count += 1
        if total > 2.5: over25_count += 1
            
    return {
        'att_avg': np.mean(scored),
        'def_avg': np.mean(conceded),
        'avg_total': np.mean(total_goals_list),
        'btts_rate': (btts_count / len(df_hist)) * 100,
        'over25_rate': (over25_count / len(df_hist)) * 100,
        'consistency': np.std(total_goals_list) # ส่วนเบี่ยงเบนมาตรฐาน (ยิ่งน้อยยิ่งนิ่ง)
    }

def calculate_confidence(h_stats, a_stats):
    # คะแนนพื้นฐานเริ่มที่ 60% (เพราะผ่านเกณฑ์คัดกรองเบื้องต้นมาแล้ว)
    score = 60.0
    
    # 1. พลังบุกรวม (Combined Attack)
    comb_att = h_stats['att_avg'] + a_stats['att_avg']
    if comb_att >= 4.0: score += 10
    elif comb_att >= 3.5: score += 7
    elif comb_att >= 3.0: score += 4
    
    # 2. เปอร์เซ็นต์ยิงทั้งคู่ (BTTS Consistency)
    avg_btts = (h_stats['btts_rate'] + a_stats['btts_rate']) / 2
    if avg_btts >= 80: score += 10
    elif avg_btts >= 70: score += 7
    elif avg_btts >= 60: score += 4
    
    # 3. การเสียประตู (Defensive Leak) - ยิ่งรั่วยิ่งดีสำหรับ OVER
    comb_def = h_stats['def_avg'] + a_stats['def_avg']
    if comb_def >= 3.5: score += 8
    elif comb_def >= 2.8: score += 5
    
    # 4. หักคะแนนถ้ามีความผันผวนสูง (Inconsistency Penalty)
    chaos = (h_stats['consistency'] + a_stats['consistency']) / 2
    if chaos > 1.8: score -= 5 # ผีเข้าผีออกเกินไป

    return min(max(score, 60), 99) # ตันที่ 99% และไม่ต่ำกว่า 60%

def analyze_match(h_stats, a_stats):
    if not h_stats or not a_stats: return None, 0
    
    # FILTER ขั้นต่ำ (ต้องผ่านตรงนี้ก่อนถึงจะคำนวณคะแนน)
    # 1. ทั้งคู่ต้องมีค่าเฉลี่ยยิง + เสีย รวมกันเกิน 2.5
    if (h_stats['avg_total'] > 2.5) and (a_stats['avg_total'] > 2.5):
        # 2. ต้องมีการยิงได้สม่ำเสมอ (BTTS Rate รวมกันต้องไม่แย่)
        if (h_stats['btts_rate'] + a_stats['btts_rate']) >= 100: # เฉลี่ยคนละ 50%
            
            conf = calculate_confidence(h_stats, a_stats)
            return "OVER", conf
            
    return None, 0

# =========================================================
# 3. 📊 BACKTESTING (5% INTERVAL BREAKDOWN)
# =========================================================
print("\n⏳ Running Backtest (Split by 5% Confidence Intervals)...")

results = [] # เก็บผลลัพธ์ [Confidence, Actual_Total_Goals]

test_data = full_df.tail(2000).reset_index(drop=True)

for i in range(len(test_data)):
    row = test_data.iloc[i]
    current_date = row['Date']
    h_team, a_team = row['HomeTeam'], row['AwayTeam']
    actual_goals = row['FTHG'] + row['FTAG']
    
    # ย้อนอดีต
    past = full_df[full_df['Date'] < current_date]
    h_form = past[(past['HomeTeam'] == h_team) | (past['AwayTeam'] == h_team)].tail(6)
    a_form = past[(past['HomeTeam'] == a_team) | (past['AwayTeam'] == a_team)].tail(6)
    
    if len(h_form) < 6 or len(a_form) < 6: continue
    
    h_s = calculate_stats(h_form, h_team)
    a_s = calculate_stats(a_form, a_team)
    
    dec, conf = analyze_match(h_s, a_s)
    
    if dec == "OVER":
        results.append({'conf': conf, 'goals': actual_goals})

# --- แสดงผลตาราง ---
print(f"\n{'='*85}")
print(f"🔬 V17.5 INTERVAL ANALYSIS (Confidence vs Reality)")
print(f"{'='*85}")
print(f"{'Range (%)':<12} | {'Matches':<8} | {'Win (3+)':<10} | {'Push (2)':<10} | {'Win Rate':<10} | {'Safe Rate':<10}")
print("-" * 85)

# สร้างช่วง 5% (60-65, 65-70, ..., 95-100)
bins = list(range(60, 101, 5))

for i in range(len(bins)-1):
    low = bins[i]
    high = bins[i+1]
    
    # กรองข้อมูลที่อยู่ในช่วงนี้
    batch = [r for r in results if low <= r['conf'] < high]
    total = len(batch)
    
    if total > 0:
        wins = sum(1 for r in batch if r['goals'] > 2.0)
        pushes = sum(1 for r in batch if r['goals'] == 2.0)
        losses = total - wins - pushes
        
        win_rate = (wins / total) * 100
        safe_rate = ((wins + pushes) / total) * 100
        
        # ใส่เกรด
        grade = ""
        if safe_rate >= 85: grade = "🔥 VIP"
        elif safe_rate >= 80: grade = "✅ GOOD"
        elif safe_rate >= 75: grade = "🤔 FAIR"
        else: grade = "💀 RISK"
        
        print(f"{low}-{high}%       | {total:<8} | {wins:<10} | {pushes:<10} | {win_rate:.1f}%    | {safe_rate:.1f}% {grade}")
    else:
        print(f"{low}-{high}%       | 0        | 0          | 0          | -         | -")

print("-" * 85)
print("* Win Rate = ยิง 3 ลูกขึ้นไป (กินเต็ม)")
print("* Safe Rate = ยิง 2 ลูกขึ้นไป (กินเต็ม + ยก/คืนทุน)")
print(f"{'='*85}")

# =========================================================
# 4. 🏥 PREDICTION (TONIGHT)
# =========================================================
# ใส่คู่บอลคืนนี้ตรงนี้
todays_matches = [
     {"Home": "Juventus",    "Away": "Lecce"},
    {"Home": "Atalanta",    "Away": "Roma"},
    {"Home": "Lazio",       "Away": "Napoli"},

    # --- จากรูปที่ 2 ---
    {"Home": "Como",        "Away": "Udinese"},
    {"Home": "Genoa",       "Away": "Pisa"},
    {"Home": "Sassuolo",    "Away": "Parma"},

    # --- จากรูปที่ 3 ---
    {"Home": "Fiorentina",  "Away": "Cremonese"},
    {"Home": "Verona",      "Away": "Torino"},
    {"Home": "Inter",       "Away": "Bologna"},
]

print("\n🏥 PREDICTION RESULT (TONIGHT V17.5)")
print(f"{'MATCH':<35} | {'CONFIDENCE':<15} | {'RATING'}")
print("-" * 85)

for match in todays_matches:
    h, a = match['Home'], match['Away']
    
    # ใช้ Data ล่าสุดทั้งหมด
    h_form = full_df[(full_df['HomeTeam'] == h) | (full_df['AwayTeam'] == h)].tail(6)
    a_form = full_df[(full_df['HomeTeam'] == a) | (full_df['AwayTeam'] == a)].tail(6)
    
    if len(h_form) < 3 or len(a_form) < 3: continue

    h_s = calculate_stats(h_form, h)
    a_s = calculate_stats(a_form, a)
    
    dec, conf = analyze_match(h_s, a_s)
    
    if dec == "OVER":
        stars = "⭐⭐"
        if conf >= 85: stars = "⭐⭐⭐⭐⭐ (VIP)"
        elif conf >= 80: stars = "⭐⭐⭐⭐ (Solid)"
        elif conf >= 75: stars = "⭐⭐⭐ (Good)"
        
        print(f"{h} vs {a:<18} | {conf:.1f}%         | {stars}")

print("-" * 85)