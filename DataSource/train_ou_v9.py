import pandas as pd
import numpy as np
import requests
from io import StringIO
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

print("⚖️ INITIATING V11.0: THE REALITY CHECK (Monotonic & ROI)...")

# =========================================================
# 1. โหลดข้อมูล
# =========================================================
urls = [
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2021/E0.csv"
]

dfs = []
for url in urls:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.content.decode('latin-1')))
            df.columns = df.columns.str.strip()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                # ต้องการราคา Odds ด้วยเพื่อคำนวณกำไร
                needed = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','HS','AS','HST','AST','BbAv>2.5','BbAv<2.5']
                available = [c for c in needed if c in df.columns]
                dfs.append(df[available])
    except: pass

full_data = pd.concat(dfs).sort_values('Date').reset_index(drop=True)
full_data = full_data.dropna()
print(f"📚 Data Loaded: {len(full_data)} matches")

# =========================================================
# 2. 🧠 LOGICAL FEATURE ENGINEERING
# =========================================================
# เราจะสร้าง Feature ที่มีความสัมพันธ์ตรงไปตรงมา (Monotonic)
feature_cols = [
    'Att_Intensity',    # (1) ยิ่งเยอะ ยิ่งน่า Over
    'Def_Fragility',    # (1) ยิ่งเยอะ ยิ่งน่า Over
    'Shot_Efficiency',  # (-1) ถ้าคมเกินไป (Overperform) แนวโน้มจะฝืด (Mean Reversion)
    'Market_Expectation'# (1) ราคาบ่อน (Implied Prob)
]

def analyze_match_logic(row, df):
    past = df[df['Date'] < row['Date']]
    h_past = past[(past['HomeTeam'] == row['HomeTeam']) | (past['AwayTeam'] == row['HomeTeam'])].tail(6)
    a_past = past[(past['HomeTeam'] == row['AwayTeam']) | (past['AwayTeam'] == row['AwayTeam'])].tail(6)
    
    if len(h_past) < 4 or len(a_past) < 4:
        return pd.Series([np.nan]*4, index=feature_cols)

    def get_stats(matches, team):
        # รวมสถิติเหย้าเยือน
        is_h = matches['HomeTeam'] == team
        g = np.where(is_h, matches['FTHG'], matches['FTAG'])
        con = np.where(is_h, matches['FTAG'], matches['FTHG'])
        sot = np.where(is_h, matches['HST'], matches['AST'])
        return np.sum(g), np.sum(con), np.sum(sot)

    h_g, h_con, h_sot = get_stats(h_past, row['HomeTeam'])
    a_g, a_con, a_sot = get_stats(a_past, row['AwayTeam'])
    
    # 1. Attack Intensity: พลังบุกรวม (วัดจากยิงเข้ากรอบ)
    att_intensity = (h_sot + a_sot) / 12.0 # เฉลี่ยต่อนัด (ประมาณ)
    
    # 2. Def Fragility: ความรั่วรวม (วัดจากเสียประตู)
    def_fragility = (h_con + a_con) / 12.0
    
    # 3. Efficiency Anomaly: ความคมที่เกินจริง
    # ถ้าค่านี้สูงมาก แปลว่ายิงเข้าง่ายเกินไป (Luck) เดี๋ยวจะยิงไม่ได้
    total_goals = h_g + a_g
    total_sot = h_sot + a_sot
    efficiency = total_goals / total_sot if total_sot > 0 else 0
    
    # 4. Market: บ่อนมองว่ายังไง (แปลง Odds เป็น %)
    odds = row.get('BbAv>2.5', 2.0)
    market_prob = (1/odds)*100
    
    return pd.Series([att_intensity, def_fragility, efficiency, market_prob], index=feature_cols)

print("⚙️ Processing V11 Logic...")
feat_df = full_data.apply(lambda x: analyze_match_logic(x, full_data), axis=1, result_type='expand')
full_data = pd.concat([full_data, feat_df], axis=1).dropna()

# =========================================================
# 3. CONSTRAINED TRAINING (บังคับ Logic)
# =========================================================
X = full_data[feature_cols]
y = ((full_data['FTHG'] + full_data['FTAG']) > 2.5).astype(int)

# Split (Last 20% for testing to simulate reality)
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
odds_test = full_data['BbAv>2.5'].iloc[split:] # เก็บราคาไว้คิดกำไร

print(f"🥊 Training with Constraints on {len(X_train)} matches...")

# 🔒 Monotonic Constraints:
# 1 = Increasing (ค่ามาก -> โอกาส Over มาก)
# -1 = Decreasing (ค่ามาก -> โอกาส Over น้อย)
# 0 = No constraint
constraints = (1, 1, -1, 1) 
# Att_Intensity (+), Def_Fragility (+), Efficiency (- ยิ่งคมเวอร์ยิ่งน่ากลัวว่าจะฝืด), Market (+)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=2,            # 🌲 Depth 2 พอ! (Stump) ให้ดูแค่ภาพกว้าง ห้ามจำรายละเอียด
    learning_rate=0.03,
    monotone_constraints=constraints, # 👈 หัวใจของ V11
    subsample=0.8,
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# 4. PROFIT/LOSS EVALUATION (ของจริง)
# =========================================================
probs = model.predict_proba(X_test)[:, 1]

print(f"\n{'='*75}")
print(f"💰 V11.0 PROFITABILITY TEST (เดิมพัน 100 บาท/คู่)")
print(f"{'='*75}")
print(f"{'Conf >':<8} | {'แมตช์':<6} | {'ถูก':<6} | {'แม่นยำ %':<10} | {'กำไร/ขาดทุน'}")
print("-" * 75)

thresholds = [0.52, 0.54, 0.56, 0.58, 0.60] # ช่วงราคา Over ปกติจะอยู่แถวๆ Prob 50-55%

for t in thresholds:
    # เลือกเล่นเฉพาะคู่ที่ Prob > Threshold
    idx = np.where(probs >= t)[0]
    
    if len(idx) > 0:
        # ดึงราคา Odds ของคู่ที่เล่น
        selected_odds = odds_test.iloc[idx].values
        selected_results = y_test.iloc[idx].values # 1=Over, 0=Under
        
        # คำนวณ P/L
        # ถ้าถูก: ได้เงิน (Odds - 1) * 100
        # ถ้าผิด: เสีย 100
        profit = np.sum(np.where(selected_results == 1, (selected_odds - 1) * 100, -100))
        
        acc = (np.sum(selected_results) / len(idx)) * 100
        
        status = "🟢 บวกรวย" if profit > 0 else "🔴 ลบยับ"
        print(f"> {t*100:.0f}%   | {len(idx):<6} | {np.sum(selected_results):<6} | {acc:.2f}%     | {profit:>6.0f} บาท  {status}")
    else:
        print(f"> {t*100:.0f}%   | 0      | -      | -          | 0")

# Save Model
joblib.dump(model, 'model_v11_reality.pkl')
print(f"\n💾 Saved V11.0: Logic-Locked Model")