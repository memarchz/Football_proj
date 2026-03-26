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
# 1. โหลดข้อมูล (Big 5 + European Leagues)
# =========================================================
print("🚀 เริ่มระบบ V5.0 EUROPE FULL OPTION (Big 5 + Extra Leagues)...")

# เพิ่มรหัสลีก: 
# P1=โปรตุเกส, N1=ฮอลแลนด์, B1=เบลเยี่ยม, G1=กรีซ, T1=ตุรกี
leagues = ['E0', 'SP1', 'D1', 'I1', 'F1', 'P1', 'N1', 'B1', 'G1', 'T1']
seasons = ['2425', '2324', '2223', '2122'] 
base_url = "https://www.football-data.co.uk/mmz4281/"

dfs = []
count = 0
for season in seasons:
    for league in leagues:
        url = f"{base_url}{season}/{league}.csv"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                csv_data = StringIO(response.content.decode('latin-1'))
                df = pd.read_csv(csv_data)
                df.columns = df.columns.str.strip()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                    df = df.dropna(subset=['Date'])
                    # เลือกเฉพาะคอลัมน์มาตรฐานที่ทุกลีกต้องมี
                    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'AvgH', 'AvgD', 'AvgA']
                    valid_cols = [c for c in cols if c in df.columns]
                    # ต้องมีราคาต่อรองครบถึงจะเอามาใช้
                    if len(valid_cols) > 5 and 'AvgH' in df.columns:
                        dfs.append(df[valid_cols])
                        count += 1
                        print(f"✅ Loaded: {league} Season {season}")
        except: pass

print(f"📚 รวมข้อมูลทั้งหมด {count} ไฟล์ลีก (ครอบคลุมทั่วยุโรป)...")
full_data = pd.concat(dfs, ignore_index=True)
full_data = full_data.sort_values(by='Date').reset_index(drop=True)

# =========================================================
# 2. Feature Engineering (Sniper Logic)
# =========================================================
print("⏳ กำลังคำนวณค่าพลังทีม (ขั้นตอนนี้จะนานขึ้นเล็กน้อย)...")

def get_sniper_stats(row, full_data):
    past = full_data[full_data['Date'] < row['Date']]
    
    # Market Impulse
    imp_h = (1 / row['AvgH']) * 100 if row['AvgH'] > 0 else 0
    imp_a = (1 / row['AvgA']) * 100 if row['AvgA'] > 0 else 0
    
    if len(past) < 100: return pd.Series([0]*10)
    
    # ค่าเฉลี่ยมาตรฐาน (ใช้ทั้งยุโรป)
    avg_h_league = past['FTHG'].mean()
    avg_a_league = past['FTAG'].mean()
    
    h_games = past[past['HomeTeam'] == row['HomeTeam']].tail(10)
    a_games = past[past['AwayTeam'] == row['AwayTeam']].tail(10)
    
    if len(h_games) < 3 or len(a_games) < 3: xg_stats = [0, 0, 0]
    else:
        h_att = h_games['FTHG'].mean() / avg_h_league if avg_h_league > 0 else 1
        h_def = h_games['FTAG'].mean() / avg_a_league if avg_a_league > 0 else 1
        a_att = a_games['FTAG'].mean() / avg_a_league if avg_a_league > 0 else 1
        a_def = a_games['FTHG'].mean() / avg_h_league if avg_h_league > 0 else 1
        h_xg = h_att * a_def * avg_h_league
        a_xg = a_att * h_def * avg_a_league
        xg_stats = [h_xg, a_xg, h_xg - a_xg]

    # H2H
    h2h_matches = past[((past['HomeTeam'] == row['HomeTeam']) & (past['AwayTeam'] == row['AwayTeam'])) | 
                        ((past['HomeTeam'] == row['AwayTeam']) & (past['AwayTeam'] == row['HomeTeam']))].tail(6)
    
    h2h_points = 0
    if len(h2h_matches) > 0:
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == row['HomeTeam']:
                h2h_points += 3 if match['FTR'] == 'H' else (1 if match['FTR'] == 'D' else 0)
            elif match['AwayTeam'] == row['HomeTeam']:
                h2h_points += 3 if match['FTR'] == 'A' else (1 if match['FTR'] == 'D' else 0)
        h2h_score = h2h_points / len(h2h_matches)
    else: h2h_score = 1.0

    # Strict Form
    sp_h = past[past['HomeTeam'] == row['HomeTeam']].tail(6)
    home_strict = sum([3 if m['FTR']=='H' else (1 if m['FTR']=='D' else 0) for _, m in sp_h.iterrows()])
    sp_a = past[past['AwayTeam'] == row['AwayTeam']].tail(6)
    away_strict = sum([3 if m['FTR']=='A' else (1 if m['FTR']=='D' else 0) for _, m in sp_a.iterrows()])

    # Overall Form
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
# 3. Training & Evaluation
# =========================================================
features = ['AvgH', 'AvgD', 'AvgA'] + cols_new
X = final_df[features]
y = LabelEncoder().fit_transform(final_df['FTR'])

train_size = int(len(X) * 0.90)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_goals = final_df.iloc[train_size:][['FTHG', 'FTAG']].values

print(f"🧠 กำลังเทรนโมเดล... (Train Data: {len(X_train)} matches)")

clf1 = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.01, max_depth=5, gamma=1.2, random_state=42)
clf2 = RandomForestClassifier(n_estimators=700, max_depth=7, random_state=42)
clf3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))

eclf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3)], voting='soft', weights=[2, 1, 1])
eclf.fit(X_train, y_train)

# Report
probs = eclf.predict_proba(X_test)
actuals = y_test

print(f"\n{'='*130}")
print(f"🔬 REPORT: ความแม่นยำโมเดล (รวมลีกรองยุโรปแล้ว)")
print(f"{'-'*130}")
print(f"{'Conf. Range':<12} | {'Matches':<8} | {'Win%':<8} | {'DC%':<16} | {'Safe% (1G)':<16} | {'Grade':<10}")
print(f"{'-'*130}")

bins = [(0.40, 0.50), (0.50, 0.55),
    (0.55, 0.60), (0.60, 0.65), 
    (0.65, 0.70), (0.70, 0.75), (0.75, 0.80),(0.80, 0.85),(0.85, 1.00),   
]

for low, high in bins:
    total, correct, dc_correct, safe_count = 0, 0, 0, 0
    for i, p in enumerate(probs):
        conf = np.max(p)
        if low <= conf < high:
            total += 1
            pred_idx = np.argmax(p)
            real_res = actuals[i]
            hg, ag = test_goals[i]
            
            if pred_idx == real_res: correct += 1; safe_count += 1
            elif abs(hg - ag) <= 1: safe_count += 1
            
            if pred_idx == 2 and real_res in [2, 1]: dc_correct += 1
            elif pred_idx == 0 and real_res in [0, 1]: dc_correct += 1
            elif pred_idx == 1 and real_res == 1: dc_correct += 1

    if total > 0:
        win_acc = (correct/total)*100
        grade = "✅ PLAY" if win_acc >= 60 else ("🔥 SUPER" if win_acc >= 75 else "🤔 RISKY")
        print(f"{low*100:.0f}-{high*100:.0f}%      | {total:<8} | {win_acc:.1f}%   | {(dc_correct/total)*100:.1f}%          | {(safe_count/total)*100:.1f}%          | {grade}")
    else:
        print(f"{low*100:.0f}-{high*100:.0f}%      | 0        | N/A      | N/A              | N/A              | -")

print(f"{'='*130}")

joblib.dump(eclf, 'model_uefa_sniper.pkl')
full_data.to_pickle('europe_db.pkl')
print(f"\n💾 บันทึกโมเดลและ Database (รวมลีกโปรตุเกส, ฮอลแลนด์, ตุรกี, กรีซ) เรียบร้อย!")
