import pandas as pd
import numpy as np
import requests
from io import StringIO
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# =========================================================
# 1. โหลดข้อมูล (5 ปี)
# =========================================================
print("🌌 เริ่มระบบ V8.0 DIMENSION BREAKER: League One...")

urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E2.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E2.csv"
]

dfs = []
for url in urls:
    try:
        response = requests.get(url, timeout=15)
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
print(f"📊 รวมข้อมูล: {len(full_data)} แมตช์")

# =========================================================
# 2. Feature Engineering (Elo + Slope + FATIGUE + VOLATILITY)
# =========================================================
print("⏳ คำนวณมิติเวลา (Fatigue) และความนิ่ง (Volatility)...")

# --- Elo Logic ---
elo_ratings = {} 
current_elos = {'Home': [], 'Away': [], 'Diff': []}

def get_elo(team): return elo_ratings.get(team, 1500)
def update_elo(home, away, result):
    k = 30
    rh, ra = get_elo(home), get_elo(away)
    exp_h = 1 / (1 + 10**((ra - rh)/400))
    elo_ratings[home] = rh + k * (result - exp_h)
    elo_ratings[away] = ra + k * ((1-result) - (1-exp_h))

for idx, row in full_data.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    current_elos['Home'].append(get_elo(h))
    current_elos['Away'].append(get_elo(a))
    current_elos['Diff'].append(get_elo(h) - get_elo(a))
    res = 1 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0)
    update_elo(h, a, res)

full_data['Elo_H'] = current_elos['Home']
full_data['Elo_A'] = current_elos['Away']
full_data['Elo_Diff'] = current_elos['Diff']

# --- The Dimension Breaker Logic ---
def get_dimension_stats(row, full_data):
    past = full_data[full_data['Date'] < row['Date']]
    
    if len(past) < 50: return pd.Series([0]*9)

    # 1. Fatigue (วันพัก)
    # หาแมตช์ล่าสุดของแต่ละทีม
    h_last = past[(past['HomeTeam'] == row['HomeTeam']) | (past['AwayTeam'] == row['HomeTeam'])].tail(1)
    a_last = past[(past['HomeTeam'] == row['AwayTeam']) | (past['AwayTeam'] == row['AwayTeam'])].tail(1)
    
    rest_h = 7 # ค่า Default
    rest_a = 7
    
    if not h_last.empty:
        rest_h = (row['Date'] - h_last.iloc[0]['Date']).days
    if not a_last.empty:
        rest_a = (row['Date'] - a_last.iloc[0]['Date']).days
        
    # ตัด Rest ให้อยู่ในช่วง 0-14 วัน (เกินนี้ถือว่าฟิตเต็มที่)
    rest_h = min(rest_h, 14)
    rest_a = min(rest_a, 14)
    rest_diff = rest_h - rest_a # ถ้าบวก แปลว่าเจ้าบ้านพักมาเยอะกว่า (ได้เปรียบ)

    # 2. Volatility (ความผันผวนของฟอร์ม 5 นัดหลัง)
    h_games = past[past['HomeTeam'] == row['HomeTeam']].tail(5)
    a_games = past[past['AwayTeam'] == row['AwayTeam']].tail(5)
    
    if len(h_games) < 5 or len(a_games) < 5: return pd.Series([0]*9)

    # คำนวณ Standard Deviation ของ Goal Difference
    h_gd = (h_games['FTHG'] - h_games['FTAG'])
    a_gd = (a_games['FTAG'] - a_games['FTHG'])
    
    h_volatility = h_gd.std() # ยิ่งเยอะ ยิ่งผีเข้าผีออก
    a_volatility = a_gd.std()
    
    # 3. Slope & Poisson (ของเดิมจาก V7)
    x = np.array([1, 2, 3, 4, 5])
    h_slope = np.polyfit(x, h_gd.values, 1)[0]
    a_slope = np.polyfit(x, a_gd.values, 1)[0]
    
    avg_h = past['FTHG'].mean()
    avg_a = past['FTAG'].mean()
    h_att = (h_games['FTHG'].mean() / avg_h) if avg_h > 0 else 1
    a_att = (a_games['FTAG'].mean() / avg_a) if avg_a > 0 else 1
    h_def = (h_games['FTAG'].mean() / avg_a) if avg_a > 0 else 1
    a_def = (a_games['FTHG'].mean() / avg_h) if avg_h > 0 else 1
    
    exp_h = h_att * a_def * avg_h
    exp_a = a_att * h_def * avg_a
    
    mkt_h = 1/row['AvgH'] if row['AvgH'] > 0 else 0
    mkt_a = 1/row['AvgA'] if row['AvgA'] > 0 else 0
    
    return pd.Series([rest_diff, h_volatility, a_volatility, h_slope, a_slope, exp_h, exp_a, mkt_h, mkt_a])

cols_dim = ['Rest_Diff', 'Vol_H', 'Vol_A', 'Slope_H', 'Slope_A', 'Exp_H', 'Exp_A', 'Mkt_H', 'Mkt_A']
full_data[cols_dim] = full_data.apply(lambda x: get_dimension_stats(x, full_data), axis=1)

final_df = full_data[full_data['Exp_H'] > 0].copy().dropna()

# =========================================================
# 3. Training (V8.0 with Context)
# =========================================================
features = ['Elo_H', 'Elo_A', 'Elo_Diff'] + cols_dim
X = final_df[features]
y = LabelEncoder().fit_transform(final_df['FTR'])

train_size = int(len(X) * 0.85)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"🧠 กำลังเทรน V8.0 (เพิ่มมิติเวลาและความเสถียร)...")

# XGBoost: ปรับ Gamma สูงขึ้น เพื่อกรองทีมที่ผันผวนออก
clf1 = xgb.XGBClassifier(n_estimators=1200, learning_rate=0.01, max_depth=5, gamma=2.5, subsample=0.75, colsample_bytree=0.8, random_state=42)

# Random Forest: เน้นจำนวนต้นไม้
clf2 = RandomForestClassifier(n_estimators=600, max_depth=9, min_samples_split=4, random_state=42)

# Logistic: C ต่ำลง เพื่อ Conservative ขึ้น
clf3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.3))

eclf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3)], voting='soft', weights=[3, 1, 1])
eclf.fit(X_train, y_train)

# =========================================================
# 4. Evaluation
# =========================================================
probs = eclf.predict_proba(X_test)
actuals = y_test

print(f"\n{'='*80}")
print(f"🌌 V8.0 DIMENSION BREAKER REPORT: LEAGUE ONE")
print(f"{'='*80}")
print(f"{'Conf. >':<8} | {'Matches':<8} | {'Win Correct':<12} | {'Win %':<8} | {'Win+Draw (DC) %':<15}")
print("-" * 80)

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

for t in thresholds:
    correct = 0; dc_correct = 0; total = 0
    for i, p in enumerate(probs):
        pred_idx = np.argmax(p)
        conf = np.max(p)
        
        if conf >= t:
            total += 1
            if pred_idx == actuals[i]: correct += 1
            if pred_idx == 2 and actuals[i] in [2, 1]: dc_correct += 1
            elif pred_idx == 0 and actuals[i] in [0, 1]: dc_correct += 1
            elif pred_idx == 1 and actuals[i] == 1: dc_correct += 1

    if total > 0:
        win_acc = (correct / total) * 100
        dc_acc = (dc_correct / total) * 100
        
        marker = ""
        if win_acc >= 80: marker = "🌌 DIMENSION BREAK"
        elif win_acc >= 70: marker = "🦄 MYTHICAL"
        elif win_acc >= 65: marker = "🔥 GOD"
        
        print(f"{t*100:.0f}%     | {total:<8} | {correct:<12} | {win_acc:.1f}%   | {dc_acc:.1f}% {marker}")

print(f"{'='*80}")
joblib.dump(eclf, 'model_v8_league1_dimension.pkl')
print("💾 บันทึกโมเดล V8.0 (ฉลาดขึ้น รู้จักความเหนื่อยและความไม่แน่นอน)")