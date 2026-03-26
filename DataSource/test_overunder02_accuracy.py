import pandas as pd
import numpy as np
import requests
from io import StringIO

print("🧪 INITIATING V7.0 BACKTESTER: TRUTH REVEALED...")

# =========================================================
# 1. โหลดข้อมูล & จัดการคอลัมน์ราคา (Auto-Map)
# =========================================================
urls = [
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
]
dfs = []

# ฟังก์ชันหาชื่อคอลัมน์ราคา
def get_col_name(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

for url in urls:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.content.decode('latin-1')))
            df.columns = df.columns.str.strip()
            
            # Map ราคา Over/Under 2.5
            col_o25 = get_col_name(df, ['Avg>2.5', 'BbAv>2.5', 'B365>2.5'])
            col_u25 = get_col_name(df, ['Avg<2.5', 'BbAv<2.5', 'B365<2.5'])
            col_h = get_col_name(df, ['AvgH', 'BbAvH', 'B365H'])
            col_a = get_col_name(df, ['AvgA', 'BbAvA', 'B365A'])
            
            if col_o25 and col_u25 and col_h and col_a:
                df['Odds_O25'] = df[col_o25]
                df['Odds_U25'] = df[col_u25]
                df['Odds_H'] = df[col_h]
                df['Odds_A'] = df[col_a]
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                dfs.append(df[['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Odds_O25','Odds_U25','Odds_H','Odds_A']].dropna())
    except: pass

full_data = pd.concat(dfs).sort_values('Date').reset_index(drop=True)
print(f"📚 Testing Data: {len(full_data)} matches")

# =========================================================
# 2. เตรียมระบบคำนวณ (Engine Setup)
# =========================================================
elo_ratings = {}
history_stats = [] # เก็บผลการทำนาย

def get_elo(team): return elo_ratings.get(team, 1500)
def update_elo(home, away, goal_h, goal_a):
    k = 30
    r_h, r_a = get_elo(home), get_elo(away)
    exp_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
    act_h = 1 if goal_h > goal_a else 0.5 if goal_h == goal_a else 0
    elo_ratings[home] = r_h + k * (act_h - exp_h)
    elo_ratings[away] = r_a + k * ((1 - act_h) - (1 - exp_h))

# ค่าเฉลี่ยลีก (ใช้ทั้ง Dataset เพื่อความง่ายในการ Test)
avg_h_league = full_data['FTHG'].mean()
avg_a_league = full_data['FTAG'].mean()

# =========================================================
# 3. เริ่มการทดสอบย้อนหลัง (Loop Matches)
# =========================================================
print("⏳ Running Simulation (This may take 20-30s)...")

for idx, row in full_data.iterrows():
    h_team, a_team = row['HomeTeam'], row['AwayTeam']
    match_date = row['Date']
    
    # --- 1. ดึงข้อมูลย้อนหลัง (ไม่รวมนัดปัจจุบัน) ---
    # เพื่อความเร็วในการ Test เราจะใช้ Elo ที่อัปเดตมาเรื่อยๆ
    # แต่ Poisson/Form จะคำนวณแบบ Simplified เพื่อให้รันจบไว
    
    # Elo (Current State)
    elo_h = get_elo(h_team)
    elo_a = get_elo(a_team)
    
    # Form (Last 5 matches lookup - Simplified logic)
    # (ในโค้ดจริงจะดึง Dataframe แต่ตรงนี้ขอจำลองค่า Form เพื่อความเร็ว หรือข้ามส่วน Form ถ้า Dataframe ใหญ่เกิน)
    # *เพื่อให้แม่นยำที่สุด เราจะใช้ Logic Elo + Odds + Poisson เป็นหลักในการ Test*
    
    # Poisson Logic
    # (จำลองค่า xG จาก Elo + League Avg เพื่อความเร็วในการ Loop 1000+ นัด)
    # xG Home ประมาณการจาก Elo diff
    elo_diff = elo_h - elo_a
    expected_xg_h = avg_h_league + (elo_diff / 500)
    expected_xg_a = avg_a_league - (elo_diff / 500)
    total_xg = expected_xg_h + expected_xg_a
    
    # --- 2. ตัดสินใจ (The Decision) ---
    decision = "PASS"
    bet_odds = 0.0
    
    # Rules จาก V7.0
    # HOME WIN
    if (elo_h > elo_a + 50) and (total_xg > 2.0) and (row['Odds_H'] < 2.10):
        decision = "HOME WIN"
        bet_odds = row['Odds_H']
        
    # AWAY WIN
    elif (elo_a > elo_h + 50) and (row['Odds_A'] < 2.10):
        decision = "AWAY WIN"
        bet_odds = row['Odds_A']

    # OVER 2.5
    elif (total_xg >= 2.80) and (row['Odds_O25'] < 1.85):
        decision = "OVER 2.5"
        bet_odds = row['Odds_O25']

    # UNDER 2.5
    elif (total_xg <= 2.30) and (row['Odds_U25'] < 1.90):
        decision = "UNDER 2.5"
        bet_odds = row['Odds_U25']

    # --- 3. ตรวจคำตอบ (Check Result) ---
    actual_goals = row['FTHG'] + row['FTAG']
    is_win = False
    
    if decision == "HOME WIN":
        is_win = (row['FTHG'] > row['FTAG'])
    elif decision == "AWAY WIN":
        is_win = (row['FTAG'] > row['FTHG'])
    elif decision == "OVER 2.5":
        is_win = (actual_goals > 2.5)
    elif decision == "UNDER 2.5":
        is_win = (actual_goals < 2.5)
    
    if decision != "PASS":
        # Profit Calculation (Unit 100)
        pnl = (bet_odds - 1) * 100 if is_win else -100
        history_stats.append({
            'Decision': decision,
            'Result': 'WIN' if is_win else 'LOSS',
            'Odds': bet_odds,
            'PnL': pnl
        })
    
    # Update Elo สำหรับนัดถัดไป
    update_elo(h_team, a_team, row['FTHG'], row['FTAG'])

# =========================================================
# 4. สรุปผล (Evaluation Report)
# =========================================================
results_df = pd.DataFrame(history_stats)

if len(results_df) > 0:
    print(f"\n{'='*65}")
    print(f"📊 V7.0 PERFORMANCE REPORT ({len(full_data)} Matches Analyzed)")
    print(f"{'='*65}")
    print(f"{'DECISION TYPE':<12} | {'PICKS':<5} | {'WIN %':<8} | {'AVG ODDS':<8} | {'NET PROFIT'}")
    print(f"{'-'*65}")

    summary = results_df.groupby('Decision').agg(
        Picks=('Result', 'count'),
        Wins=('Result', lambda x: (x=='WIN').sum()),
        AvgOdds=('Odds', 'mean'),
        NetProfit=('PnL', 'sum')
    )
    
    summary['WinRate'] = (summary['Wins'] / summary['Picks']) * 100
    
    for decision, row in summary.iterrows():
        profit_str = f"{row['NetProfit']:+.0f}"
        color = "🟢" if row['NetProfit'] > 0 else "🔴"
        print(f"{decision:<12} | {row['Picks']:<5} | {row['WinRate']:>6.1f}% | {row['AvgOdds']:>8.2f} | {profit_str:>8} {color}")

    print(f"{'-'*65}")
    total_picks = summary['Picks'].sum()
    total_profit = summary['NetProfit'].sum()
    print(f"TOTAL RESULT | {total_picks:<5} | {(results_df['Result']=='WIN').mean()*100:>6.1f}% |    -     | {total_profit:+.0f} THB")
    print(f"{'='*65}")

else:
    print("❌ No matches matched the strict criteria.")