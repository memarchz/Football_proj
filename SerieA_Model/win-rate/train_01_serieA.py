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
print("MODE: Time-Split + Market Logic...")

urls = [
    "https://www.football-data.co.uk/mmz4281/2526/I1.csv", # ฤดูกาลปัจจุบัน
    "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2223/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2122/I1.csv"
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
# 2. Feature Engineering
# =========================================================
print("⏳ กำลังคำนวณฟีเจอร์ใหม่: Implied Probability + Strict Form...")

def get_sniper_stats(row, full_data):
    past = full_data[full_data['Date'] < row['Date']]
    
    # Implied Probability
    imp_h = (1 / row['AvgH']) * 100 if row['AvgH'] > 0 else 0
    imp_a = (1 / row['AvgA']) * 100 if row['AvgA'] > 0 else 0
    
    if len(past) < 50: return pd.Series([0, 0, 0, 0, 0, 0, imp_h, imp_a])
    
    # xG Calculation
    avg_h = past['FTHG'].mean()
    avg_a = past['FTAG'].mean()
    h_games = past[past['HomeTeam'] == row['HomeTeam']].tail(10)
    a_games = past[past['AwayTeam'] == row['AwayTeam']].tail(10)
    
    if len(h_games) < 3 or len(a_games) < 3: 
        xg_stats = [0, 0, 0]
    else:
        h_att = h_games['FTHG'].mean() / avg_h if avg_h > 0 else 1
        h_def = h_games['FTAG'].mean() / avg_a if avg_a > 0 else 1
        a_att = a_games['FTAG'].mean() / avg_a if avg_a > 0 else 1
        a_def = a_games['FTHG'].mean() / avg_h if avg_h > 0 else 1
        h_xg = h_att * a_def * avg_h
        a_xg = a_att * h_def * avg_a
        xg_stats = [h_xg, a_xg, h_xg - a_xg]

    # H2H
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
    else:
        h2h_score = 1.0

    # Strict Form
    sp_h = past[past['HomeTeam'] == row['HomeTeam']].tail(6)
    home_form = sum([3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0) for _, m in sp_h.iterrows()])

    sp_a = past[past['AwayTeam'] == row['AwayTeam']].tail(6)
    away_form = sum([3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0) for _, m in sp_a.iterrows()])

    return pd.Series(xg_stats + [h2h_score, home_form, away_form, imp_h, imp_a])

# Apply Features
cols_new = ['Home_xG', 'Away_xG', 'xG_Diff', 'H2H', 'Home_Form', 'Away_Form', 'Imp_H', 'Imp_A']
full_data[cols_new] = full_data.apply(lambda x: get_sniper_stats(x, full_data), axis=1)

final_df = full_data[full_data['Home_xG'] > 0].copy().dropna()

# =========================================================
# 3. Training
# =========================================================
features = ['AvgH', 'AvgD', 'AvgA'] + cols_new
X = final_df[features]
y = LabelEncoder().fit_transform(final_df['FTR']) # A=0, D=1, H=2

train_size = int(len(X) * 0.85)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ดึงข้อมูลประตูจริงสำหรับชุด Test
test_goals = final_df.iloc[train_size:][['FTHG', 'FTAG']].values

print(f"🧠 กำลังเทรน V4.0 (Train: {len(X_train)} | Test: {len(X_test)})...")

clf1 = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, gamma=1.5, min_child_weight=3, random_state=42)
clf2 = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=5, random_state=42)
clf3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))

eclf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3)], voting='soft', weights=[2, 1, 1])
eclf.fit(X_train, y_train)

# =========================================================
# 4. Evaluation (Updated Display)
# =========================================================
probs = eclf.predict_proba(X_test)
actuals = y_test

print(f"\n{'='*125}")
print(f"🎯 V4.0 PERFORMANCE REPORT (Win vs Double Chance vs Safe Zone)")
print(f"{'='*125}")
# เพิ่มคอลัมน์ Draw
print(f"{'Conf. >':<8} | {'Matches':<8} | {'Correct':<8} | {'Wrong':<8} | {'Wrong (1G)':<10} | {'Draw':<6} | {'Win %':<8} | {'DC %':<8} | {'Safe %':<8}")
print("-" * 125)

thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

for t in thresholds:
    correct = 0         
    wrong_count = 0     
    lose_1g_count = 0   
    draw_count = 0      # นับจำนวนเสมอ
    dc_correct = 0  
    safe_count = 0      
    total = 0
    
    for i, p in enumerate(probs):
        pred_idx = np.argmax(p) 
        conf = np.max(p)        
        
        if conf >= t:
            total += 1
            hg, ag = test_goals[i]
            goal_diff = abs(hg - ag)
            
            # Check Draw (LabelEncoder: A=0, D=1, H=2)
            if actuals[i] == 1:
                draw_count += 1

            # 1. Check Win Correct & Safe Logic
            if pred_idx == actuals[i]: 
                correct += 1
                safe_count += 1 
            else:
                wrong_count += 1
                if goal_diff == 1:
                    lose_1g_count += 1
                    safe_count += 1 
                elif goal_diff == 0:
                    safe_count += 1 
            
            # 2. Check Double Chance
            if pred_idx == 2: 
                if actuals[i] in [2, 1]: dc_correct += 1
            elif pred_idx == 0:
                if actuals[i] in [0, 1]: dc_correct += 1
            elif pred_idx == 1:
                if actuals[i] == 1: dc_correct += 1

    if total > 0:
        win_acc = (correct / total) * 100
        dc_acc = (dc_correct / total) * 100
        safe_acc = (safe_count / total) * 100
        
        marker = ""
        if win_acc >= 70: marker = "🔥 GOD"
        elif dc_acc >= 90: marker = "🛡️ SAFE"
        elif win_acc >= 60: marker = "✅ OK"
        
        print(f"{t*100:.0f}%     | {total:<8} | {correct:<8} | {wrong_count:<8} | {lose_1g_count:<10} | {draw_count:<6} | {win_acc:.1f}%   | {dc_acc:.1f}%   | {safe_acc:.1f}% {marker}")

# =========================================================
# 🔬 V5.0 REALITY CHECK (Detailed 5% Intervals)
# =========================================================
print(f"\n{'='*130}")
print(f"🔬 V5.0 REALITY CHECK (Detailed 5% Intervals)")
print(f"{'='*130}")
# เพิ่มคอลัมน์ Draw
print(f"{'Range':<12} | {'Matches':<8} | {'Correct':<8} | {'Wrong':<8} | {'Wrong (1G)':<10} | {'Draw':<6} | {'Win %':<8} | {'DC %':<8} | {'Safe %':<8}")
print("-" * 130)

bins = [
    (0.40, 0.45), (0.45, 0.50), 
    (0.50, 0.55), (0.55, 0.60),
    (0.60, 0.65), (0.65, 0.70), 
    (0.70, 0.75), (0.75, 0.80), 
    (0.80, 1.01) 
]

for low, high in bins:
    correct = 0
    wrong_count = 0
    lose_1g_count = 0
    draw_count = 0  # นับจำนวนเสมอ
    dc_correct = 0
    safe_count = 0
    total = 0
    
    for i, p in enumerate(probs):
        conf = np.max(p)
        
        if low <= conf < high:
            total += 1
            pred_idx = np.argmax(p)
            hg, ag = test_goals[i]
            goal_diff = abs(hg - ag)
            
            # Check Draw
            if actuals[i] == 1:
                draw_count += 1
            
            # Win Correct & Safe Logic
            if pred_idx == actuals[i]: 
                correct += 1
                safe_count += 1 
            else:
                wrong_count += 1
                if goal_diff == 1:
                    lose_1g_count += 1
                    safe_count += 1 
                elif goal_diff == 0:
                    safe_count += 1 
            
            # DC Correct
            if pred_idx == 2: # Home
                if actuals[i] in [2, 1]: dc_correct += 1
            elif pred_idx == 0: # Away
                if actuals[i] in [0, 1]: dc_correct += 1
            elif pred_idx == 1: # Draw
                if actuals[i] == 1: dc_correct += 1

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
        
        print(f"{low*100:.0f}-{high*100:.0f}%      | {total:<8} | {correct:<8} | {wrong_count:<8} | {lose_1g_count:<10} | {draw_count:<6} | {win_acc:.1f}%   | {dc_acc:.1f}%   | {safe_acc:.1f}% {grade}")
    else:
        print(f"{low*100:.0f}-{high*100:.0f}%      | {0:<8} | {0:<8} | {0:<8} | {0:<10} | {0:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")

print(f"{'='*130}")

joblib.dump(eclf, 'model_v4_serieA.pkl') 
print(f"\n💾 บันทึกโมเดลเรียบร้อย!")