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
# 1. โหลดข้อมูล
# =========================================================
print("🚀 เริ่มระบบ V5.0 FULL OPTION: Strict + Momentum + Detailed Check...")

urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
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

# =========================================================
# 2. Feature Engineering (V5 Logic)
# =========================================================
print("⏳ กำลังคำนวณฟีเจอร์ใหม่...")

def get_sniper_stats(row, full_data):
    past = full_data[full_data['Date'] < row['Date']]
    imp_h = (1 / row['AvgH']) * 100 if row['AvgH'] > 0 else 0
    imp_a = (1 / row['AvgA']) * 100 if row['AvgA'] > 0 else 0
    
    if len(past) < 50: return pd.Series([0]*10)
    
    avg_h = past['FTHG'].mean()
    avg_a = past['FTAG'].mean()
    h_games = past[past['HomeTeam'] == row['HomeTeam']].tail(10)
    a_games = past[past['AwayTeam'] == row['AwayTeam']].tail(10)
    
    if len(h_games) < 3 or len(a_games) < 3: xg_stats = [0, 0, 0]
    else:
        h_att = h_games['FTHG'].mean() / avg_h if avg_h > 0 else 1
        h_def = h_games['FTAG'].mean() / avg_a if avg_a > 0 else 1
        a_att = a_games['FTAG'].mean() / avg_a if avg_a > 0 else 1
        a_def = a_games['FTHG'].mean() / avg_h if avg_h > 0 else 1
        h_xg = h_att * a_def * avg_h
        a_xg = a_att * h_def * avg_a
        xg_stats = [h_xg, a_xg, h_xg - a_xg]

    h2h_matches = past[((past['HomeTeam'] == row['HomeTeam']) & (past['AwayTeam'] == row['AwayTeam'])) | 
                        ((past['HomeTeam'] == row['AwayTeam']) & (past['AwayTeam'] == row['HomeTeam']))].tail(6)
    
    h2h_points = 0
    if len(h2h_matches) > 0:
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == row['HomeTeam']:
                if match['FTR'] == 'H': h2h_points += 3
                elif match['FTR'] == 'D': h2h_points += 1
            elif match['AwayTeam'] == row['HomeTeam']:
                if match['FTR'] == 'A': h2h_points += 3
                elif match['FTR'] == 'D': h2h_points += 1
        h2h_score = h2h_points / len(h2h_matches)
    else: h2h_score = 1.0

    sp_h = past[past['HomeTeam'] == row['HomeTeam']].tail(6)
    home_strict = sum([3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0) for _, m in sp_h.iterrows()])
    sp_a = past[past['AwayTeam'] == row['AwayTeam']].tail(6)
    away_strict = sum([3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0) for _, m in sp_a.iterrows()])

    last5_h = past[(past['HomeTeam'] == row['HomeTeam']) | (past['AwayTeam'] == row['HomeTeam'])].tail(5)
    h_overall = 0
    for _, m in last5_h.iterrows():
        if m['HomeTeam'] == row['HomeTeam']: h_overall += 3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0)
        else: h_overall += 3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0)

    last5_a = past[(past['HomeTeam'] == row['AwayTeam']) | (past['AwayTeam'] == row['AwayTeam'])].tail(5)
    a_overall = 0
    for _, m in last5_a.iterrows():
        if m['HomeTeam'] == row['AwayTeam']: a_overall += 3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0)
        else: a_overall += 3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0)

    return pd.Series(xg_stats + [h2h_score, home_strict, away_strict, h_overall, a_overall, imp_h, imp_a])

cols_new = ['Home_xG', 'Away_xG', 'xG_Diff', 'H2H', 'Home_Strict', 'Away_Strict', 'Home_Overall', 'Away_Overall', 'Imp_H', 'Imp_A']
full_data[cols_new] = full_data.apply(lambda x: get_sniper_stats(x, full_data), axis=1)

final_df = full_data[full_data['Home_xG'] > 0].copy().dropna()

# =========================================================
# 3. Training & Prep Evaluation
# =========================================================
features = ['AvgH', 'AvgD', 'AvgA'] + cols_new
X = final_df[features]
y = LabelEncoder().fit_transform(final_df['FTR'])

# Split & Keep Goals for Detailed Report
train_size = int(len(X) * 0.85)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_goals = final_df.iloc[train_size:][['FTHG', 'FTAG']].values  # เก็บสกอร์จริงไว้เช็ค 1G Loss

print(f"🧠 กำลังเทรน V5.0 (Train: {len(X_train)} | Test: {len(X_test)})...")

clf1 = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, gamma=1.5, min_child_weight=3, random_state=42)
clf2 = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=5, random_state=42)
clf3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))

eclf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3)], voting='soft', weights=[2, 1, 1])
eclf.fit(X_train, y_train)

# =========================================================
# 4. Evaluation (ตารางแบบละเอียด)
# =========================================================
probs = eclf.predict_proba(X_test)
actuals = y_test

print(f"\n{'='*130}")
print(f"🔬 V5.0 REALITY CHECK (Detailed 5% Intervals)")
print(f"{'Range':<12} | {'Matches':<8} | {'Correct':<8} | {'Wrong':<8} | {'Wrong (1G)':<10} | {'Draw':<6} | {'Win %':<8} | {'DC %':<8} | {'Safe %':<8}")
print(f"{'-'*130}")

bins = [(0.38,0.40),
    (0.40, 0.45), (0.45, 0.50), 
    (0.50, 0.55), (0.55, 0.60),
    (0.60, 0.65), (0.65, 0.70), 
    (0.70, 0.75), (0.75, 0.80), 
    (0.80, 1.01) 
]

for low, high in bins:
    correct = 0
    wrong = 0
    wrong_1g = 0
    draw_count = 0 
    dc_correct = 0
    safe_count = 0
    total = 0
    
    for i, p in enumerate(probs):
        conf = np.max(p)
        
        if low <= conf < high:
            total += 1
            pred_idx = np.argmax(p)
            real_res = actuals[i]
            hg, ag = test_goals[i]
            goal_diff = abs(hg - ag)
            
            # นับจำนวนเสมอ
            if real_res == 1: draw_count += 1
            
            # 1. เช็คความถูกต้อง (Win %)
            if pred_idx == real_res:
                correct += 1
                safe_count += 1
            else:
                wrong += 1
                # เช็คผิดแบบ 1 ลูก หรือ ผิดแต่เสมอ (Safe Zone)
                if goal_diff <= 1: 
                    wrong_1g += 1
                    safe_count += 1 # ถือว่า Safe ถ้าแพ้ลูกเดียวหรือเสมอ
            
            # 2. เช็ค Double Chance (DC %)
            if pred_idx == 2: # ทายเจ้าบ้าน
                if real_res in [2, 1]: dc_correct += 1
            elif pred_idx == 0: # ทายทีมเยือน
                if real_res in [0, 1]: dc_correct += 1
            elif pred_idx == 1: # ทายเสมอ
                if real_res == 1: dc_correct += 1

    if total > 0:
        win_acc = (correct / total) * 100
        dc_acc = (dc_correct / total) * 100
        safe_acc = (safe_count / total) * 100
        
        grade = ""
        if win_acc < 50: grade = "💀 DANGER"
        elif win_acc < 60: grade = "😐 RISKY"
        elif win_acc < 65: grade = "🤔 FAIR"
        elif win_acc < 75: grade = "✅ GOOD"
        elif win_acc < 85: grade = "💎 SOLID"
        else: grade = "🔥 GOD"
        
        print(f"{low*100:.0f}-{high*100:.0f}%      | {total:<8} | {correct:<8} | {wrong:<8} | {wrong_1g:<10} | {draw_count:<6} | {win_acc:.1f}%   | {dc_acc:.1f}%   | {safe_acc:.1f}% {grade}")
    else:
        print(f"{low*100:.0f}-{high*100:.0f}%      | {0:<8} | {0:<8} | {0:<8} | {0:<10} | {0:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")

print(f"{'='*130}")

joblib.dump(eclf, 'model_v5_sniper.pkl')
print(f"\n💾 บันทึกโมเดล V5.0 และตารางวิเคราะห์เรียบร้อย!")