import pandas as pd
import numpy as np
import requests
from io import StringIO
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

print("⚖️ INITIATING V11.1: THE REALITY CHECK (Auto-Detect Odds)...")

# =========================================================
# 1. โหลดข้อมูล & แก้ปัญหาชื่อคอลัมน์อัตโนมัติ
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
            df.columns = df.columns.str.strip() # ลบช่องว่างหัวตาราง
            
            # 🔍 Auto-Detect Odds Column (แก้จุด Error)
            # หาคอลัมน์ราคา Over 2.5 ที่มีอยู่จริง (ลำดับความสำคัญ: Avg -> BbAv -> B365)
            # เราจะวนลูปหาว่าไฟล์นี้ใช้ชื่ออะไร
            possible_names = ['Avg>2.5', 'BbAv>2.5', 'B365>2.5']
            over_col = next((c for c in possible_names if c in df.columns), None)
            
            if over_col and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                
                # เปลี่ยนชื่อให้เป็นมาตรฐานเดียวกัน 'Odds_Over' เพื่อไม่ให้ Error อีก
                df['Odds_Over'] = df[over_col]
                
                # เลือกเฉพาะคอลัมน์ที่ต้องใช้
                needed = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','HS','AS','HST','AST','Odds_Over']
                available = [c for c in needed if c in df.columns]
                
                # ต้องมีข้อมูลประตูและราคาครบ
                temp_df = df[available].dropna()
                dfs.append(temp_df)
    except: pass

full_data = pd.concat(dfs).sort_values('Date').reset_index(drop=True)
print(f"📚 Data Loaded: {len(full_data)} matches")

# =========================================================
# 2. 🧠 LOGICAL FEATURE ENGINEERING
# =========================================================
feature_cols = [
    'Att_Intensity',    # (1) พลังบุก
    'Def_Fragility',    # (1) ความรั่ว
    'Shot_Efficiency',  # (-1) ความคมเกินจริง
    'Market_Expectation'# (1) ความคาดหวังตลาด
]

def analyze_match_logic(row, df):
    past = df[df['Date'] < row['Date']]
    h_past = past[(past['HomeTeam'] == row['HomeTeam']) | (past['AwayTeam'] == row['HomeTeam'])].tail(6)
    a_past = past[(past['HomeTeam'] == row['AwayTeam']) | (past['AwayTeam'] == row['AwayTeam'])].tail(6)
    
    if len(h_past) < 4 or len(a_past) < 4:
        return pd.Series([np.nan]*4, index=feature_cols)

    def get_stats(matches, team):
        is_h = matches['HomeTeam'] == team
        g = np.where(is_h, matches['FTHG'], matches['FTAG'])
        con = np.where(is_h, matches['FTAG'], matches['FTHG'])
        sot = np.where(is_h, matches['HST'], matches['AST'])
        return np.sum(g), np.sum(con), np.sum(sot)

    h_g, h_con, h_sot = get_stats(h_past, row['HomeTeam'])
    a_g, a_con, a_sot = get_stats(a_past, row['AwayTeam'])
    
    # Logic Calculation
    att_intensity = (h_sot + a_sot) / 12.0
    def_fragility = (h_con + a_con) / 12.0
    
    total_goals = h_g + a_g
    total_sot = h_sot + a_sot
    # ป้องกันหาร 0
    efficiency = total_goals / total_sot if total_sot > 0 else 0
    
    # ใช้คอลัมน์ Odds_Over ที่เรา rename มาแล้ว (ไม่มีพลาด)
    odds = row['Odds_Over']
    market_prob = (1/odds)*100 if odds > 0 else 50
    
    return pd.Series([att_intensity, def_fragility, efficiency, market_prob], index=feature_cols)

print("⚙️ Processing V11 Logic...")
# ใช้ result_type='expand' เพื่อความชัวร์
feat_df = full_data.apply(lambda x: analyze_match_logic(x, full_data), axis=1, result_type='expand')

# รวมร่างและตัด NaN
full_data = pd.concat([full_data, feat_df], axis=1).dropna()
print(f"✅ Ready for Training: {len(full_data)} matches")

# =========================================================
# 3. CONSTRAINED TRAINING
# =========================================================
X = full_data[feature_cols]
y = ((full_data['FTHG'] + full_data['FTAG']) > 2.5).astype(int)

# Split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
odds_test = full_data['Odds_Over'].iloc[split:] # ใช้ชื่อใหม่

print(f"🥊 Training XGBoost on {len(X_train)} matches...")

# Constraints: Att(+), Def(+), Eff(-), Market(+)
constraints = (1, 1, -1, 1) 

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=2,
    learning_rate=0.03,
    monotone_constraints=constraints,
    subsample=0.8,
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# 4. PROFIT/LOSS EVALUATION
# =========================================================
if len(X_test) > 0:
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*75}")
    print(f"💰 V11.1 PROFITABILITY TEST (Unit: 100 THB)")
    print(f"{'='*75}")
    print(f"{'Conf >':<8} | {'Match':<6} | {'Win':<6} | {'Acc %':<8} | {'Profit (THB)'}")
    print("-" * 75)

    thresholds = [0.52, 0.54, 0.56, 0.58, 0.60]

    for t in thresholds:
        idx = np.where(probs >= t)[0]
        if len(idx) > 0:
            selected_odds = odds_test.iloc[idx].values
            selected_results = y_test.iloc[idx].values
            
            # Profit Calculation: (Odds - 1) * Stake - Stake(if lost)
            # P/L = (Win * (Odds-1)) - (Loss * 1)
            profit = 0
            for i in range(len(idx)):
                if selected_results[i] == 1:
                    profit += (selected_odds[i] - 1) * 100
                else:
                    profit -= 100
            
            acc = (np.sum(selected_results) / len(idx)) * 100
            status = "🟢 PROFIT" if profit > 0 else "🔴 LOSS"
            print(f"> {t*100:.0f}%   | {len(idx):<6} | {np.sum(selected_results):<6} | {acc:.2f}%   | {profit:>6.0f} {status}")
        else:
            print(f"> {t*100:.0f}%   | 0      | -      | -        | 0")
else:
    print("❌ Not enough data for testing.")

joblib.dump(model, 'model_v11_reality.pkl')
print(f"\n💾 Saved V11.1 Model Successfully")