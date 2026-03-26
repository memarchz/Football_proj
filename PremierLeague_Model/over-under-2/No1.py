import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings

# ปิด Warning
warnings.filterwarnings('ignore')

# =========================================================
# 1. LOAD DATA & SETUP
# =========================================================
print("🔄 V17.8 (Odds Weighted): Loading Data & Calibrating System...")

urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
]

dfs = []
for url in urls:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            csv_data = StringIO(r.content.decode('latin-1'))
            df = pd.read_csv(csv_data)
            df.columns = df.columns.str.strip()
            
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365>2.5']
            existing_cols = [c for c in cols if c in df.columns]
            
            if len(existing_cols) >= 5:
                dfs.append(df[existing_cols])
    except: pass

if not dfs:
    print("❌ Error: โหลดข้อมูลไม่ได้"); exit()

full_df = pd.concat(dfs, ignore_index=True)
full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')

# แปลงราคาเป็นตัวเลข
if 'B365>2.5' in full_df.columns:
    full_df['B365>2.5'] = pd.to_numeric(full_df['B365>2.5'], errors='coerce')
else:
    full_df['B365>2.5'] = np.nan

full_df = full_df.dropna(subset=['Date', 'FTHG', 'FTAG']).sort_values('Date').reset_index(drop=True)
print(f"✅ Loaded {len(full_df)} matches.")

# =========================================================
# 2. 🛠️ CORE LOGIC (ODDS WEIGHTED)
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
        'consistency': np.std(total_goals_list)
    }

# [UPDATED] รับค่า odds เข้ามาคำนวณคะแนน
def calculate_confidence(h_stats, a_stats, odds=None):
    score = 60.0
    
    # 1. Combined Attack
    comb_att = h_stats['att_avg'] + a_stats['att_avg']
    if comb_att >= 4.0: score += 10
    elif comb_att >= 3.5: score += 7
    elif comb_att >= 3.0: score += 4
    
    # 2. BTTS Consistency
    avg_btts = (h_stats['btts_rate'] + a_stats['btts_rate']) / 2
    if avg_btts >= 80: score += 10
    elif avg_btts >= 70: score += 7
    elif avg_btts >= 60: score += 4
    
    # 3. Defensive Leak
    comb_def = h_stats['def_avg'] + a_stats['def_avg']
    if comb_def >= 3.5: score += 8
    elif comb_def >= 2.8: score += 5
    
    # 4. Inconsistency Penalty
    chaos = (h_stats['consistency'] + a_stats['consistency']) / 2
    if chaos > 1.8: score -= 5

    # 5. [NEW] Odds Weighting (Bookmaker Factor)
    # ถ้าราคาไหลไปทางต่ำ (เจ้ามือกลัว) -> บวกคะแนน
    # ถ้าราคาไหลไปทางสูง (เจ้ามือท้า) -> ลบคะแนน
    if odds is not None and odds > 0:
        if odds <= 1.50: score += 8    # มั่นใจมาก (1.50 ลงมา)
        elif odds <= 1.65: score += 5  # มั่นใจ (1.51 - 1.65)
        elif odds <= 1.75: score += 2  # ค่อนข้างมั่นใจ
        elif odds >= 2.00: score -= 5  # เสี่ยง (2.00 ขึ้นไป)
        elif odds >= 2.20: score -= 8  # เสี่ยงมาก

    return min(max(score, 60), 99)

def analyze_match(h_stats, a_stats, odds=None):
    if not h_stats or not a_stats: return None, 0
    if (h_stats['avg_total'] > 2.5) and (a_stats['avg_total'] > 2.5):
        if (h_stats['btts_rate'] + a_stats['btts_rate']) >= 100:
            # ส่ง odds ไปคำนวณด้วย
            conf = calculate_confidence(h_stats, a_stats, odds)
            return "OVER", conf
    return None, 0

# =========================================================
# 3. 📊 BACKTESTING
# =========================================================
print("\n⏳ Running Backtest (Stats + Odds Integration)...")

results = []
test_data = full_df.tail(2000).reset_index(drop=True)

for i in range(len(test_data)):
    row = test_data.iloc[i]
    current_date = row['Date']
    h_team, a_team = row['HomeTeam'], row['AwayTeam']
    actual_goals = row['FTHG'] + row['FTAG']
    odds = row['B365>2.5'] if 'B365>2.5' in row and not pd.isna(row['B365>2.5']) else 0
    
    past = full_df[full_df['Date'] < current_date]
    h_form = past[(past['HomeTeam'] == h_team) | (past['AwayTeam'] == h_team)].tail(6)
    a_form = past[(past['HomeTeam'] == a_team) | (past['AwayTeam'] == a_team)].tail(6)
    
    if len(h_form) < 6 or len(a_form) < 6: continue
    
    h_s = calculate_stats(h_form, h_team)
    a_s = calculate_stats(a_form, a_team)
    
    # ส่ง odds เข้าไปในระบบวิเคราะห์เลย
    dec, conf = analyze_match(h_s, a_s, odds)
    
    if dec == "OVER":
        results.append({'conf': conf, 'goals': actual_goals, 'odds': odds})

# --- แสดงผลตาราง (Hybrid) ---
print(f"\n{'='*115}")
print(f"🔬 V17.8 INTELLIGENT ANALYSIS (Stats + Bookie Odds Integration)")
print(f"{'='*115}")
print(f"{'Range':<8} | {'Mat':<5} | {'Win(3+)':<7} | {'Push(2)':<7} | {'Win%':<6} | {'Safe%':<6} || {'AvgOdd':<6} | {'ROI%':<7} | {'Grade'}")
print("-" * 115)

bins = list(range(60, 101, 5))

for i in range(len(bins)-1):
    low = bins[i]
    high = bins[i+1]
    
    batch = [r for r in results if low <= r['conf'] < high]
    total = len(batch)
    
    if total > 0:
        wins = sum(1 for r in batch if r['goals'] > 2.5)
        pushes = sum(1 for r in batch if r['goals'] == 2.0)
        
        win_rate = (wins / total) * 100
        safe_rate = ((wins + pushes) / total) * 100
        
        valid_odds_batch = [r for r in batch if r['odds'] > 0]
        if valid_odds_batch:
            total_invested = len(valid_odds_batch)
            total_return = sum(r['odds'] for r in valid_odds_batch if r['goals'] > 2.5)
            profit = total_return - total_invested
            roi = (profit / total_invested) * 100
            avg_odds = sum(r['odds'] for r in valid_odds_batch) / len(valid_odds_batch)
        else:
            roi = 0; avg_odds = 0

        grade = ""
        if safe_rate >= 85: grade = "🔥 VIP"
        elif safe_rate >= 80: grade = "✅ GOOD"
        elif safe_rate >= 75: grade = "🤔 FAIR"
        else: grade = "💀 RISK"
        
        print(f"{low}-{high}% | {total:<5} | {wins:<7} | {pushes:<7} | {win_rate:.1f}%  | {safe_rate:.1f}%  || {avg_odds:.2f}   | {roi:+.1f}%   | {grade}")
    else:
        print(f"{low}-{high}% | 0     | 0       | 0       | -      | -      || -      | -       | -")

print("-" * 115)
print(f"{'='*115}")

# =========================================================
# 4. 🏥 PREDICTION (TONIGHT)
# =========================================================
todays_matches = [
     {"Home": "Bournemouth",    "Away": "Tottenham",     "Odds": 1.57},
    {"Home": "Brentford",      "Away": "Sunderland",    "Odds": 1.72},
    {"Home": "Everton",        "Away": "Wolves",        "Odds": 2.10},
    {"Home": "Man City",       "Away": "Brighton",      "Odds": 1.44},
    {"Home": "Crystal Palace", "Away": "Aston Villa",   "Odds": 1.85},
    {"Home": "Fulham",         "Away": "Chelsea",       "Odds": 1.66},
    {"Home": "Burnley",        "Away": "Man United",    "Odds": 1.75},
    {"Home": "Newcastle",      "Away": "Leeds",         "Odds": 1.60}
]

print("\n🏥 PREDICTION RESULT (TONIGHT V17.8)")
print(f"{'MATCH':<30} | {'CONF %':<8} | {'ODDS':<6} | {'VALUE?':<10} | {'RATING'}")
print("-" * 100)

for match in todays_matches:
    h, a = match['Home'], match['Away']
    odds = match.get('Odds', 0)
    
    h_form = full_df[(full_df['HomeTeam'] == h) | (full_df['AwayTeam'] == h)].tail(6)
    a_form = full_df[(full_df['HomeTeam'] == a) | (full_df['AwayTeam'] == a)].tail(6)
    
    if len(h_form) < 3 or len(a_form) < 3: continue

    h_s = calculate_stats(h_form, h)
    a_s = calculate_stats(a_form, a)
    
    # ส่ง odds ไปคำนวณร่วมด้วย
    dec, conf = analyze_match(h_s, a_s, odds)
    
    if dec == "OVER":
        stars = "⭐⭐"
        if conf >= 85: stars = "⭐⭐⭐⭐⭐ (VIP)"
        elif conf >= 80: stars = "⭐⭐⭐⭐ (Solid)"
        elif conf >= 75: stars = "⭐⭐⭐ (Good)"
        
        # ปรับเกณฑ์ Value เล็กน้อย เพราะคะแนน Conf ถูกปรับด้วย Odds มาแล้วระดับหนึ่ง
        value_msg = "-"
        if odds > 0:
            implied_prob = (1 / odds) * 100
            if conf > implied_prob: value_msg = "💎 YES" # แค่มากกว่าก็ถือว่าดีแล้ว เพราะ Conf กรองมาเข้ม
            else: value_msg = "⚖️ FAIR"
        
        print(f"{h} vs {a:<13} | {conf:.1f}%   | {odds:.2f}   | {value_msg:<10} | {stars}")

print("-" * 100)