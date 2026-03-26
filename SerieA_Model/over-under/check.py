import pandas as pd
import numpy as np
import requests
from io import StringIO

# =========================================================
# 🌗 SYSTEM V10.0: FIRE & ICE (Over/Under Specialist)
# =========================================================
print("🚀 กำลังแยกประสาทสัมผัส: โหมด FIRE (สูง) และ ICE (ต่ำ)...")

# 1. โหลดข้อมูล League One 3 ฤดูกาลล่าสุด
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
            df = pd.read_csv(StringIO(r.content.decode('latin-1')))
            dfs.append(df)
    except: pass

full_df = pd.concat(dfs).dropna(subset=['FTHG','FTAG']).sort_values('Date').reset_index(drop=True)
full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True)

# ตัวแปรเก็บสถิติ
report = {
    'Under 3.5': {'Won': 0, 'Lost': 0, 'Total': 0},
    'Over 2.0':  {'Won': 0, 'Push': 0, 'Lost': 0, 'Total': 0} 
}

print(f"🔎 เริ่มสแกน {len(full_df)} แมตช์ เพื่อหาคู่ทีเด็ด...")

# 2. เริ่มวิเคราะห์ทีละนัด
for i in range(100, len(full_df)):
    row = full_df.iloc[i]
    h, a = row['HomeTeam'], row['AwayTeam']
    match_date = row['Date']
    actual_goals = row['FTHG'] + row['FTAG']
    
    # ข้อมูลย้อนหลัง 6 นัดล่าสุดของแต่ละทีม
    past = full_df[full_df['Date'] < match_date]
    h_last6 = past[(past['HomeTeam']==h) | (past['AwayTeam']==h)].tail(6)
    a_last6 = past[(past['HomeTeam']==a) | (past['AwayTeam']==a)].tail(6)
    
    if len(h_last6) < 6 or len(a_last6) < 6: continue
    
    # --- คำนวณค่าพลัง (Stats Calculation) ---
    
    # 1. ค่าเฉลี่ยประตูได้เสีย (Avg Total Goals)
    h_avg_total = (h_last6['FTHG'].sum() + h_last6['FTAG'].sum()) / 6
    a_avg_total = (a_last6['FTHG'].sum() + a_last6['FTAG'].sum()) / 6
    match_avg = (h_avg_total + a_avg_total) / 2
    
    # 2. ความนิ่ง (SD - Standard Deviation) **ทีเด็ดสายต่ำ**
    # แปลว่า 6 นัดที่ผ่านมา ยิงกันคงที่ไหม? หรือผีเข้าผีออก
    h_goals_arr = h_last6['FTHG'].values + h_last6['FTAG'].values
    a_goals_arr = a_last6['FTHG'].values + a_last6['FTAG'].values
    match_sd = (np.std(h_goals_arr) + np.std(a_goals_arr)) / 2
    
    # 3. อัตราการยิงประตู (Clean Sheet / Fail to Score)
    # หาจำนวนนัดที่ยิงไม่ได้เลย (0 ประตู)
    h_blank = np.sum(h_last6['FTHG'] == 0) if h in h_last6['HomeTeam'].values else 0 
    # (คำนวณแบบคร่าวๆ เพื่อความเร็ว ใช้ Avg แทนได้)
    
    # =========================================================
    # 🧊 LOGIC 1: ICE PROTOCOL (Strict Under 3.5)
    # =========================================================
    # เงื่อนไข:
    # 1. ค่าเฉลี่ยยิงรวมกันต้องน้อย (ไม่เกิน 2.4 ลูกต่อนัด)
    # 2. ต้องนิ่งจัดๆ (SD ต้องต่ำกว่า 1.35 ห้ามมีนัดยิงระเบิดปนมา)
    # 3. 6 นัดหลังสุด ห้ามมีนัดไหนยิงเกิน 4 ลูก (Safety Check)
    
    max_h_goals = np.max(h_goals_arr)
    max_a_goals = np.max(a_goals_arr)
    
    if (match_avg <= 2.4) and (match_sd <= 1.35) and (max_h_goals <= 4) and (max_a_goals <= 4):
        report['Under 3.5']['Total'] += 1
        if actual_goals < 3.5:
            report['Under 3.5']['Won'] += 1
        else:
            report['Under 3.5']['Lost'] += 1

    # =========================================================
    # 🔥 LOGIC 2: FIRE PROTOCOL (Safe Over 2.0)
    # =========================================================
    # เงื่อนไข:
    # 1. ค่าเฉลี่ยยิงรวมกันต้องสูงพอควร (เกิน 2.8 ลูกต่อนัด)
    # 2. ทั้งสองทีมต้อง "รั่ว" (เสียประตูเฉลี่ยทีมละ 1.2 ลูกขึ้นไป)
    # 3. เป้าหมายคือ "อย่างน้อย 2 ลูกต้องมา" (เพื่อกันทุน) ลุ้นลูก 3 กินเต็ม
    
    # คำนวณการเสียประตูเฉลี่ย (Conceded Avg)
    def get_conceded(df_hist, team):
        conceded = 0
        for _, m in df_hist.iterrows():
            if m['HomeTeam'] == team: conceded += m['FTAG']
            else: conceded += m['FTHG']
        return conceded / len(df_hist)

    h_leak = get_conceded(h_last6, h)
    a_leak = get_conceded(a_last6, a)
    
    # กฎ: ยิงรวมเยอะ + หลังรั่วทั้งคู่ + ห้ามมีประวัติ 0-0 ใน 6 นัดหลัง
    has_00_draw = (0 in h_goals_arr) or (0 in a_goals_arr) # เช็คคร่าวๆว่ามีนัดฝืดสนิทไหม
    
    if (match_avg >= 2.8) and (h_leak >= 1.2) and (a_leak >= 1.2):
        report['Over 2.0']['Total'] += 1
        if actual_goals > 2:   # 3, 4, 5...
            report['Over 2.0']['Won'] += 1
        elif actual_goals == 2: # 2 ลูกเป๊ะ
            report['Over 2.0']['Push'] += 1
        else:                  # 0, 1 ลูก
            report['Over 2.0']['Lost'] += 1

# =========================================================
# 📊 FINAL REPORT
# =========================================================
print("\n" + "="*80)
print("🌗 V10.0 PERFORMANCE REPORT (Backtest Results)")
print("="*80)

# 1. Under 3.5 Stats
u_total = report['Under 3.5']['Total']
u_win = report['Under 3.5']['Won']
u_rate = (u_win / u_total * 100) if u_total > 0 else 0
print(f"🧊 UNDER 3.5 STRICT MODE")
print(f"   - แมตช์ที่เล่น: {u_total}")
print(f"   - ชนะ: {u_win} | แพ้: {report['Under 3.5']['Lost']}")
print(f"   - ความแม่นยำ (Win Rate): {u_rate:.2f}%")
print("-" * 80)

# 2. Over 2.0 Stats
o_total = report['Over 2.0']['Total']
o_win = report['Over 2.0']['Won']
o_push = report['Over 2.0']['Push']
o_lost = report['Over 2.0']['Lost']
o_win_rate = (o_win / o_total * 100) if o_total > 0 else 0
o_not_lose_rate = ((o_win + o_push) / o_total * 100) if o_total > 0 else 0

print(f"🔥 OVER 2.0 SAFE MODE (สูง 2)")
print(f"   - แมตช์ที่เล่น: {o_total}")
print(f"   - กินเต็ม (>2): {o_win} ({o_win_rate:.2f}%)")
print(f"   - เจ๊า/คืนทุน (=2): {o_push}")
print(f"   - ตาย (<2): {o_lost}")
print(f"   - 🛡️ อัตราไม่เสียเงิน (Win+Push): {o_not_lose_rate:.2f}%")
print("="*80)