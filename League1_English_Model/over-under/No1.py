import pandas as pd
import numpy as np
import requests
from io import StringIO

# =========================================================
# 🔪 V16.0 BATCH PREDICTOR: THE SURGEON (Full + Detailed Report)
# =========================================================
print("🔄 กำลังโหลดข้อมูลสถิติจาก E1, E2, E3 และ EC (National League)...")

# 1. โหลดข้อมูล
urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E1.csv", # Championship
    "https://www.football-data.co.uk/mmz4281/2526/E2.csv", # League One
    "https://www.football-data.co.uk/mmz4281/2526/E3.csv", # League Two
    "https://www.football-data.co.uk/mmz4281/2526/EC.csv", # National League
    "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E3.csv",
    "https://www.football-data.co.uk/mmz4281/2425/EC.csv"
]

dfs = []
for url in urls:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.content.decode('latin-1')))
            dfs.append(df)
    except: pass

if not dfs:
    print("❌ โหลดข้อมูลไม่สำเร็จ กรุณาเช็คอินเทอร์เน็ต")
    exit()

full_df = pd.concat(dfs).dropna(subset=['FTHG','FTAG']).sort_values('Date').reset_index(drop=True)
full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True)

# =========================================================
# 🛠️ CORE LOGIC FUNCTIONS (V16)
# =========================================================

def calculate_stats(df_hist, team_name):
    if len(df_hist) < 1: return None
    scored = []
    fts = 0
    btts_count = 0
    total_goals = 0
    for _, m in df_hist.iterrows():
        g_for = m['FTHG'] if m['HomeTeam'] == team_name else m['FTAG']
        g_ag = m['FTAG'] if m['HomeTeam'] == team_name else m['FTHG']
        scored.append(g_for)
        total_goals += (g_for + g_ag)
        if g_for == 0: fts += 1
        if (g_for > 0) and (g_ag > 0): btts_count += 1
        
    return {
        'att_avg': np.mean(scored),
        'max_scored': np.max(scored),
        'fts': fts,
        'btts': btts_count,
        'avg_total': total_goals / len(df_hist)
    }

def check_v16_criteria(h_stats, a_stats):
    if not h_stats or not a_stats: return None

    # 🧊 1. UNDER 3.5 Check
    if (h_stats['att_avg'] < 1.1) and (a_stats['att_avg'] < 1.1) and \
       (h_stats['max_scored'] <= 2) and (a_stats['max_scored'] <= 2) and \
       ((h_stats['fts'] + a_stats['fts']) >= 3):
        return "UNDER"

    # 🔥 2. OVER 2.0 Check
    elif (h_stats['btts'] >= 3) and (a_stats['btts'] >= 3) and \
         (h_stats['avg_total'] > 2.5) and (a_stats['avg_total'] > 2.5) and \
         (h_stats['att_avg'] >= 1.0) and (a_stats['att_avg'] >= 1.0):
        return "OVER"
        
    return None

# =========================================================
# 📊 PART 1: SYSTEM ACCURACY CHECK (DETAILED REPORT)
# =========================================================
results = {"UNDER": {"Win": 0, "Total": 0}, "OVER": {"Win": 0, "Push": 0, "Total": 0}}

print("⏳ กำลังประมวลผลสถิติย้อนหลัง (Backtesting)...")

start_idx = max(0, len(full_df) - 2000) # เช็ค 2000 นัดล่าสุดเพื่อให้ได้ Data เยอะๆ
for i in range(start_idx, len(full_df)):
    row = full_df.iloc[i]
    actual = row['FTHG'] + row['FTAG']
    h, a = row['HomeTeam'], row['AwayTeam']
    
    past = full_df[full_df['Date'] < row['Date']]
    h_last6 = past[(past['HomeTeam']==h) | (past['AwayTeam']==h)].tail(6)
    a_last6 = past[(past['HomeTeam']==a) | (past['AwayTeam']==a)].tail(6)
    
    if len(h_last6) < 6 or len(a_last6) < 6: continue
    
    dec = check_v16_criteria(calculate_stats(h_last6, h), calculate_stats(a_last6, a))
    
    if dec == "UNDER":
        results["UNDER"]["Total"] += 1
        if actual < 3.5: results["UNDER"]["Win"] += 1
            
    elif dec == "OVER":
        results["OVER"]["Total"] += 1
        if actual > 2: results["OVER"]["Win"] += 1
        elif actual == 2: results["OVER"]["Push"] += 1

# --- คำนวณผลลัพธ์ละเอียด ---
u_total = results["UNDER"]["Total"]
u_win = results["UNDER"]["Win"]
u_loss = u_total - u_win
u_acc = (u_win/u_total*100) if u_total else 0

o_total = results["OVER"]["Total"]
o_win = results["OVER"]["Win"]
o_push = results["OVER"]["Push"]
o_loss = o_total - (o_win + o_push)
o_safe = ((o_win + o_push)/o_total*100) if o_total else 0

print("\n" + "="*60)
print(f"📊 V16 SYSTEM REPORT (Last {u_total + o_total} qualified matches)")
print("-" * 60)
print(f"🧊 UNDER 3.5 SUMMARY:")
print(f"   ► ทายทั้งหมด : {u_total} คู่")
print(f"   ► ✅ ถูก (Win) : {u_win} คู่")
print(f"   ► ❌ ผิด (Loss): {u_loss} คู่")
print(f"   ► 🎯 Accuracy  : {u_acc:.2f}%")
print("-" * 60)
print(f"🔥 OVER 2.0 SUMMARY:")
print(f"   ► ทายทั้งหมด   : {o_total} คู่")
print(f"   ► ✅ ถูก (Win)   : {o_win} คู่ (ยิง 3+)")
print(f"   ► 🟡 เจ๊า (Push) : {o_push} คู่ (คืนทุน)")
print(f"   ► ❌ ผิด (Loss)  : {o_loss} คู่")
print(f"   ► 🛡️ Safe Rate  : {o_safe:.2f}% (Win + Push)")
print("="*60 + "\n")

# =========================================================
# 🔮 PART 2: PREDICTION (Tonight's Matches)
# =========================================================
matches_to_analyze = [
    {"Home": "Coventry", "Away": "Ipswich"},
    {"Home": "Stoke", "Away": "Sheffield United"},
    {"Home": "West Brom", "Away": "QPR"},
    {"Home": "Middlesbrough", "Away": "Hull"},
    {"Home": "Oxford", "Away": "Swansea"},
    {"Home": "Wrexham", "Away": "Preston"},
    {"Home": "Leicester", "Away": "Derby"},
    {"Home": "Sheffield Weds", "Away": "Blackburn"},
    {"Home": "Millwall", "Away": "Bristol City"},
    {"Home": "Norwich", "Away": "Watford"},
    {"Home": "Portsmouth", "Away": "Charlton"},
    {"Home": "Birmingham", "Away": "Southampton"},
]

print("🏥 PREDICTION REPORT (V16 Logic)")
print("-" * 60)

found = False
for match in matches_to_analyze:
    h_team, a_team = match['Home'], match['Away']
    
    h_last6 = full_df[(full_df['HomeTeam']==h_team) | (full_df['AwayTeam']==h_team)].tail(6)
    a_last6 = full_df[(full_df['HomeTeam']==a_team) | (full_df['AwayTeam']==a_team)].tail(6)
    
    if len(h_last6) < 6 or len(a_last6) < 6:
        continue

    decision = check_v16_criteria(calculate_stats(h_last6, h_team), calculate_stats(a_last6, a_team))
    
    if decision:
        found = True
        print(f"⚽ {h_team} vs {a_team}")
        if decision == "UNDER":
            print(f"   🎯 REC: 🧊 UNDER 3.5 (Acc: {u_acc:.1f}%)")
        elif decision == "OVER":
            print(f"   🎯 REC: 🔥 OVER 2.0 (Safe: {o_safe:.1f}%)")
        print("-" * 30)

if not found:
    print("🧹 ไม่พบคู่ที่เข้าเกณฑ์ V16 ในรายการวันนี้ครับ")