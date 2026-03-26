import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO

# =========================================================
# 1. โหลดโมเดลและประวัติ 5 ปี
# =========================================================
print("🌌 LOADING V8.0 DIMENSION PREDICTOR...")

try:
    model = joblib.load('model_v8_league1_dimension.pkl')
    print("✅ โหลดโมเดลสำเร็จ!")
except:
    print("❌ ไม่เจอไฟล์โมเดล! (ต้องรันโค้ด Train V8 ก่อน)"); exit()

# โหลดข้อมูล
urls = [
   "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/SC0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/SC0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/SC0.csv"
]

dfs = []
print("⏳ กำลังสร้างฐานข้อมูลมิติเวลา (Time & Volatility)...")
for url in urls:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            csv_data = StringIO(response.content.decode('latin-1'))
            df = pd.read_csv(csv_data)
            df.columns = df.columns.str.strip()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['Date'])
                cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'AvgH', 'AvgD', 'AvgA']
                valid_cols = [c for c in cols if c in df.columns]
                if len(valid_cols) > 5:
                    dfs.append(df[valid_cols])
    except: pass

full_data = pd.concat(dfs, ignore_index=True)
full_data = full_data.sort_values(by='Date').reset_index(drop=True)

# --- สร้าง Elo ล่าสุด ---
elo_ratings = {}
def get_elo(team): return elo_ratings.get(team, 1500)
def update_elo(home, away, result):
    k = 30
    rh, ra = get_elo(home), get_elo(away)
    exp_h = 1 / (1 + 10**((ra - rh)/400))
    elo_ratings[home] = rh + k * (result - exp_h)
    elo_ratings[away] = ra + k * ((1-result) - (1-exp_h))

for idx, row in full_data.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    res = 1 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0)
    update_elo(h, a, res)

print(f"📊 ฐานข้อมูลพร้อมใช้งาน: {len(full_data)} แมตช์")

# =========================================================
# 2. ฟังก์ชันคำนวณ Features V8 (ซับซ้อนที่สุด)
# =========================================================
def get_dimension_features(home, away, odds_h, odds_d, odds_a, match_date, full_data):
    # match_date: วันที่จะเตะ (ใส่เป็น string 'YYYY-MM-DD')
    target_date = pd.to_datetime(match_date)
    
    # 1. Elo
    elo_h = get_elo(home)
    elo_a = get_elo(away)
    elo_diff = elo_h - elo_a
    
    # 2. Fatigue (วันพัก)
    # หาแมตช์ล่าสุดก่อนวันแข่ง
    past = full_data[full_data['Date'] < target_date]
    
    h_last = past[(past['HomeTeam'] == home) | (past['AwayTeam'] == home)].tail(1)
    a_last = past[(past['HomeTeam'] == away) | (past['AwayTeam'] == away)].tail(1)
    
    rest_h = 7; rest_a = 7
    if not h_last.empty: rest_h = (target_date - h_last.iloc[0]['Date']).days
    if not a_last.empty: rest_a = (target_date - a_last.iloc[0]['Date']).days
    
    rest_h = min(rest_h, 14)
    rest_a = min(rest_a, 14)
    rest_diff = rest_h - rest_a

    # 3. Volatility & Slope (5 นัดล่าสุด)
    h_games = past[past['HomeTeam'] == home].tail(5)
    a_games = past[past['AwayTeam'] == away].tail(5)
    
    if len(h_games) < 5 or len(a_games) < 5:
        print(f"⚠️ ข้อมูลไม่พอวิเคราะห์ความเสถียร: {home} หรือ {away}")
        return None

    # Slope
    h_gd = (h_games['FTHG'] - h_games['FTAG']).values
    a_gd = (a_games['FTAG'] - a_games['FTHG']).values
    x = np.array([1, 2, 3, 4, 5])
    slope_h = np.polyfit(x, h_gd, 1)[0]
    slope_a = np.polyfit(x, a_gd, 1)[0]
    
    # Volatility (SD)
    vol_h = np.std(h_gd)
    vol_a = np.std(a_gd)
    
    # 4. Poisson
    avg_h = full_data['FTHG'].mean()
    avg_a = full_data['FTAG'].mean()
    
    h_hist = past[past['HomeTeam'] == home].tail(10)
    a_hist = past[past['AwayTeam'] == away].tail(10)
    
    h_att = (h_hist['FTHG'].mean()/avg_h) if avg_h>0 else 1
    a_att = (a_hist['FTAG'].mean()/avg_a) if avg_a>0 else 1
    h_def = (h_hist['FTAG'].mean()/avg_a) if avg_a>0 else 1
    a_def = (a_hist['FTHG'].mean()/avg_h) if avg_h>0 else 1
    
    exp_h = h_att * a_def * avg_h
    exp_a = a_att * h_def * avg_a
    
    # 5. Market
    mkt_h = 1/odds_h
    mkt_a = 1/odds_a
    
    # เรียงลำดับ Feature ให้ตรงกับตอน Train เป๊ะๆ
    return [elo_h, elo_a, elo_diff, rest_diff, vol_h, vol_a, slope_h, slope_a, exp_h, exp_a, mkt_h, mkt_a]

# =========================================================
# 3. โซนทำนายผล V8 (อัปเดตจากรูปภาพ)
# =========================================================
print("\n" + "="*80)
print("🌌 V8.0 DIMENSION BREAKER: PREDICTION ZONE")
print("="*80)

# 🛠️ ตั้งค่าวันที่แข่ง (จากรูปคือ 30/12)
match_date = "2025-12-31" 

# 🛠️ รายชื่อคู่บอลจากรูปภาพ (แปลงชื่อให้ตรงกับ Database)
matches = [
   {"Home": "Dundee", "Away": "Kilmarnock", "OddsH": 2.74, "OddsD": 3.125, "OddsA": 2.59}, # ชื่อใน DB อาจเป็น Dundee FC
    {"Home": "Hibernian", "Away": "Aberdeen", "OddsH": 1.85, "OddsD": 3.55, "OddsA": 4.08},
    {"Home": "Livingston", "Away": "Dundee United", "OddsH": 2.53, "OddsD": 3.24, "OddsA": 2.70},
    {"Home": "Rangers", "Away": "St Mirren", "OddsH": 1.62, "OddsD": 3.86, "OddsA": 5.22},
    {"Home": "Motherwell", "Away": "Celtic", "OddsH": 4.405, "OddsD": 3.92, "OddsA": 1.71},
]

print(f"Date: {match_date}")
print(f"{'Match':<30} | {'Pred':<8} | {'Conf %':<8} | {'Recommendation'}")
print("-" * 85)

# --- ลูปทำนายผล (เหมือนเดิม) ---
for m in matches:
    # เพิ่ม Try-Except เพื่อป้องกัน error กรณีหาชื่อทีมข้ามลีกไม่เจอ
    try:
        feats = get_dimension_features(m['Home'], m['Away'], m['OddsH'], m['OddsD'], m['OddsA'], match_date, full_data)
        
        if feats:
            cols = ['Elo_H', 'Elo_A', 'Elo_Diff', 'Rest_Diff', 'Vol_H', 'Vol_A', 'Slope_H', 'Slope_A', 'Exp_H', 'Exp_A', 'Mkt_H', 'Mkt_A']
            X_pred = pd.DataFrame([feats], columns=cols)
            
            prob = model.predict_proba(X_pred)[0]
            pred_idx = np.argmax(prob)
            conf = np.max(prob) * 100
            
            res_map = {0: 'Away', 1: 'Draw', 2: 'Home'}
            pred_text = res_map[pred_idx]
            
            rec = ""
            if conf >= 65: rec = "🦄 MYTHICAL (100% DC Record)"
            elif conf >= 60: rec = "🔥 GOD TIER (High Win%)"
            elif conf >= 55: rec = "✅ Safe Zone"
            else: rec = "⛔ PASS (Too Risky)"
            
            print(f"{m['Home']} vs {m['Away']:<15} | {pred_text:<8} | {conf:.1f}%   | {rec}")
    except Exception as e:
        print(f"{m['Home']} vs {m['Away']:<15} | N/A      | 0.0%     | ⚠️ Data Not Found (Different League)")

print("-" * 85)