import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings

# ปิด Warning สีแดงให้ตาโล่งๆ
warnings.filterwarnings('ignore')

# =========================================================
# 1. SETUP & DATA LOADING (โหลดข้อมูล E0 และ SC0)
# =========================================================
print("🔄 V18.0 THE SURGEON: Loading Premier League & Scottish Data...")

urls = [
   
    # Scotland
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
            df.columns = df.columns.str.strip() # ลบช่องว่างชื่อคอลัมน์
            
            # เลือกเฉพาะคอลัมน์ที่ต้องใช้เพื่อลดขนาด
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            if all(c in df.columns for c in cols):
                dfs.append(df[cols])
    except: pass

if not dfs:
    print("❌ Error: โหลดข้อมูลไม่ได้ (เช็คเน็ต)"); exit()

# รวมข้อมูลและจัดการวันที่
full_df = pd.concat(dfs, ignore_index=True)
full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')
full_df = full_df.dropna(subset=['Date', 'FTHG', 'FTAG', 'HomeTeam', 'AwayTeam'])
full_df = full_df.sort_values('Date').reset_index(drop=True)

print(f"✅ Loaded {len(full_df)} matches successfully.")

# =========================================================
# 2. 🛠️ CORE LOGIC V18 (Chaos Filter + Momentum)
# =========================================================

def calculate_stats_v18(df_form, team_name, side):
    """
    df_form: DataFrame 5 นัดล่าสุด (เหย้า หรือ เยือน)
    side: 'Home' หรือ 'Away'
    """
    if len(df_form) < 3: return None
    
    scored = []
    conceded = []
    total_goals_list = []
    chaos_games = 0 # นับจำนวนนัดที่ยิงกันถล่มทลาย (Over 4.5)
    
    for _, row in df_form.iterrows():
        if side == 'Home':
            g_for, g_ag = row['FTHG'], row['FTAG']
        else:
            g_for, g_ag = row['FTAG'], row['FTHG']
            
        scored.append(g_for)
        conceded.append(g_ag)
        total = g_for + g_ag
        total_goals_list.append(total)
        
        # 🚩 CHAOS FLAG: ถ้านัดไหนสกอร์รวม >= 5 ถือว่าทีมนี้อันตราย (ผีเข้าผีออก)
        if total >= 5: 
            chaos_games += 1
            
    return {
        'att_avg': np.mean(scored),      # ยิงเฉลี่ย
        'def_avg': np.mean(conceded),    # เสียเฉลี่ย
        'avg_total': np.mean(total_goals_list), # ประตูรวมเฉลี่ย
        'chaos_count': chaos_games,      # จำนวนนัดที่สกอร์หลุดโลก
        'last_match_total': total_goals_list[-1] # สกอร์รวมนัดล่าสุด
    }

def check_v18_criteria(h_stats, a_stats):
    if not h_stats or not a_stats: return None

    # คำนวณค่ารวม
    comb_att = h_stats['att_avg'] + a_stats['att_avg']
    comb_def = h_stats['def_avg'] + a_stats['def_avg']

    # 🧊 1. UNDER 3.5 (The "Stable" Game)
    # คอนเซปต์: ห้ามมีความวุ่นวาย (Chaos=0) และ เกมรุกรวมกันต้องไม่ดุ
    
    # กฏเหล็ก 1: ใน 5 นัดล่าสุด ห้ามมีนัดไหนจบสกอร์รวม 5 ลูกขึ้นไป (ตัดทีมผีเข้าทิ้ง)
    if (h_stats['chaos_count'] == 0) and (a_stats['chaos_count'] == 0):
        
        # กฏเหล็ก 2: นัดล่าสุดของทั้งคู่ ต้องไม่เพิ่งยิงกันกระจาย (Momentum Check)
        if (h_stats['last_match_total'] <= 3) and (a_stats['last_match_total'] <= 3):
            
            # กฏเหล็ก 3: พลังบุกรวมต้องต่ำ (เฉลี่ยยิงรวมกัน < 2.6)
            # เรายอมให้หลังรั่วได้นิดหน่อย แต่เกมรุกห้ามเทพ
            if comb_att < 2.6:
                 return "UNDER"

    # 🔥 2. OVER 2.0 (The Goal Machine)
    # คอนเซปต์: ถ้า Chaos เยอะ หรือ พลังบุกโหด หรือ หลังรั่วโหด -> ไป Over
    if (comb_att >= 3.0) or (comb_def >= 3.2) or (h_stats['chaos_count'] > 0 and a_stats['chaos_count'] > 0):
        if (h_stats['avg_total'] > 2.3) and (a_stats['avg_total'] > 2.3):
            return "OVER"
            
    return None

# =========================================================
# 3. 📊 BACKTESTING (วัดผลย้อนหลัง)
# =========================================================
print("\n⏳ Running Backtest V18 (Chaos Logic) on last 1500 matches...")
results = {"UNDER": {"Win": 0, "Total": 0}, "OVER": {"Win": 0, "Push": 0, "Total": 0}}

# เทสย้อนหลัง 1,500 นัดล่าสุด
test_data = full_df.tail(1500).reset_index(drop=True)

for i in range(len(test_data)):
    row = test_data.iloc[i]
    current_date = row['Date']
    h_team, a_team = row['HomeTeam'], row['AwayTeam']
    actual = row['FTHG'] + row['FTAG']
    
    # ดึงข้อมูลอดีต
    past = full_df[full_df['Date'] < current_date]
    
    # 🎯 STRICT SPLIT FORM: เหย้าดูเหย้า เยือนดูเยือน
    h_form = past[past['HomeTeam'] == h_team].tail(5)
    a_form = past[past['AwayTeam'] == a_team].tail(5)
    
    if len(h_form) < 5 or len(a_form) < 5: continue
    
    h_s = calculate_stats_v18(h_form, h_team, 'Home')
    a_s = calculate_stats_v18(a_form, a_team, 'Away')
    
    dec = check_v18_criteria(h_s, a_s)
    
    if dec == "UNDER":
        results["UNDER"]["Total"] += 1
        if actual < 3.5: results["UNDER"]["Win"] += 1
    elif dec == "OVER":
        results["OVER"]["Total"] += 1
        if actual > 2: results["OVER"]["Win"] += 1
        elif actual == 2: results["OVER"]["Push"] += 1

# แสดงผล Backtest
u_total = results["UNDER"]["Total"]
u_win = results["UNDER"]["Win"]
u_acc = (u_win / u_total * 100) if u_total > 0 else 0

o_total = results["OVER"]["Total"]
o_win = results["OVER"]["Win"]
o_push = results["OVER"]["Push"]
o_safe = ((o_win + o_push) / o_total * 100) if o_total > 0 else 0

print(f"\n{'='*60}")
print(f"📊 V18 PERFORMANCE REPORT (Chaos Filter)")
print(f"{'='*60}")
print(f"🧊 UNDER 3.5 : Acc {u_acc:.2f}% ({u_win}/{u_total})")
print(f"🔥 OVER 2.0  : Safe Rate {o_safe:.2f}% (Win {o_win} | Push {o_push} | Loss {o_total - o_win - o_push})")
print(f"{'='*60}")

# =========================================================
# 4. 🏥 PREDICTION (ใช้งานจริง)
# =========================================================
# ใส่คู่บอลที่ต้องการเช็คที่นี่
todays_matches = [
    {"Home": "Burnley",     "Away": "Newcastle"},
     {"Home": "West Ham",     "Away": "Brighton"},
     {"Home": "Nott'm Forest",     "Away": "Everton"},
     {"Home": "Chelsea",     "Away": "Bournemouth"},
     {"Home": "Arsenal",     "Away": "Aston Villa"},
     {"Home": "Man United",     "Away": "Wolves"},
]

print("\n🏥 PREDICTION RESULT (TONIGHT V18)")
print(f"{'MATCH':<35} | {'PREDICTION':<25} | {'NOTE'}")
print("-" * 85)

for match in todays_matches:
    h, a = match['Home'], match['Away']
    
    # ใช้ข้อมูลล่าสุดที่มีในไฟล์ CSV
    past = full_df
    
    h_form = past[past['HomeTeam'] == h].tail(5)
    a_form = past[past['AwayTeam'] == a].tail(5)
    
    if len(h_form) < 3 or len(a_form) < 3:
        # print(f"{h} vs {a}: ข้อมูลไม่พอ")
        continue

    h_s = calculate_stats_v18(h_form, h, 'Home')
    a_s = calculate_stats_v18(a_form, a, 'Away')
    
    dec = check_v18_criteria(h_s, a_s)
    
    if dec:
        if dec == "UNDER": 
            note = f"Chaos: {h_s['chaos_count']}/{a_s['chaos_count']} (Zero)"
            icon = "🧊 UNDER 3.5"
        else: 
            comb_att = h_s['att_avg'] + a_s['att_avg']
            note = f"Att Power: {comb_att:.1f}"
            icon = "🔥 OVER 2.0"
            
        print(f"{h} vs {a:<18} | {icon:<25} | {note}")
    else:
        # คู่ไหนไม่เข้าเกณฑ์ V18 (ไม่ชัวร์) ก็จะไม่แสดงผล
        pass

print("-" * 85)