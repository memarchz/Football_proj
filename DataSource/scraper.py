import pandas as pd
import numpy as np
import requests
from io import StringIO
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# =========================================================
# PART 1 (V3.1): โหลดข้อมูล + H2H (6 นัด) + Specific Form
# =========================================================
print("🚀 เริ่มระบบ God Mode V3.1: H2H (6 นัด) + เจาะลึกฟอร์มเหย้า/เยือน...")

urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2021/E0.csv"
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
# 🛠️ ฟังก์ชันสร้างฟีเจอร์ขั้นเทพ (ปรับ H2H เหลือ 6 นัด)
# =========================================================
print("⏳ กำลังคำนวณ: H2H (6 นัดล่าสุด), Home-Form(Home), Away-Form(Away)...")

def get_specific_stats(row, full_data):
    past = full_data[full_data['Date'] < row['Date']]
    
    # --- 1. Basic xG (10 นัดเหมือนเดิม) ---
    if len(past) < 50: return pd.Series([0, 0, 0, 0, 0, 0])
    
    avg_h = past['FTHG'].mean()
    avg_a = past['FTAG'].mean()
    h_games_all = past[past['HomeTeam'] == row['HomeTeam']].tail(10)
    a_games_all = past[past['AwayTeam'] == row['AwayTeam']].tail(10)
    
    if len(h_games_all) < 3 or len(a_games_all) < 3: 
        xg_stats = [0, 0, 0]
    else:
        h_att = h_games_all['FTHG'].mean() / avg_h if avg_h > 0 else 1
        h_def = h_games_all['FTAG'].mean() / avg_a if avg_a > 0 else 1
        a_att = a_games_all['FTAG'].mean() / avg_a if avg_a > 0 else 1
        a_def = a_games_all['FTHG'].mean() / avg_h if avg_h > 0 else 1
        h_xg = h_att * a_def * avg_h
        a_xg = a_att * h_def * avg_a
        xg_stats = [h_xg, a_xg, h_xg - a_xg]

    # --- 2. H2H Form (แก้ไข: เหลือ 6 นัดล่าสุด) ---
    h2h_matches = past[((past['HomeTeam'] == row['HomeTeam']) & (past['AwayTeam'] == row['AwayTeam'])) | 
                       ((past['HomeTeam'] == row['AwayTeam']) & (past['AwayTeam'] == row['HomeTeam']))].tail(6)
    
    h2h_points = 0
    if len(h2h_matches) > 0:
        for _, match in h2h_matches.iterrows():
            # ทีมเหย้าปัจจุบัน เป็นเจ้าบ้านในอดีต
            if match['HomeTeam'] == row['HomeTeam']:
                if match['FTR'] == 'H': h2h_points += 3
                elif match['FTR'] == 'D': h2h_points += 1
            # ทีมเหย้าปัจจุบัน เป็นทีมเยือนในอดีต
            elif match['AwayTeam'] == row['HomeTeam']:
                if match['FTR'] == 'A': h2h_points += 3
                elif match['FTR'] == 'D': h2h_points += 1
        h2h_score = h2h_points / len(h2h_matches)
    else:
        h2h_score = 1.0 # ค่ากลาง

    # --- 3. Home Form Specific (6 นัดล่าสุดในบ้าน) ---
    specific_home_games = past[past['HomeTeam'] == row['HomeTeam']].tail(6)
    home_sp_points = 0
    for _, match in specific_home_games.iterrows():
        if match['FTR'] == 'H': home_sp_points += 3
        elif match['FTR'] == 'D': home_sp_points += 1

    # --- 4. Away Form Specific (6 นัดล่าสุดนอกบ้าน) ---
    specific_away_games = past[past['AwayTeam'] == row['AwayTeam']].tail(6)
    away_sp_points = 0
    for _, match in specific_away_games.iterrows():
        if match['FTR'] == 'A': away_sp_points += 3
        elif match['FTR'] == 'D': away_sp_points += 1

    return pd.Series(xg_stats + [h2h_score, home_sp_points, away_sp_points])

# Apply ฟังก์ชัน
cols_new = ['Home_xG', 'Away_xG', 'xG_Diff', 'H2H_Avg_Points', 'Home_Home_Form', 'Away_Away_Form']
full_data[cols_new] = full_data.apply(lambda x: get_specific_stats(x, full_data), axis=1)

# กรองเฉพาะคู่ที่มีข้อมูล
final_df = full_data[full_data['Home_xG'] > 0].copy()

# บันทึก CSV
final_df.to_csv('clean_premier_league_v3.csv', index=False)
print(f"✅ บันทึกข้อมูล V3.1 เรียบร้อย ({len(final_df)} แมตช์)")

# =========================================================
# PART 2: เทรนโมเดล (9 Features)
# =========================================================
features = ['AvgH', 'AvgD', 'AvgA', 
            'Home_xG', 'Away_xG', 'xG_Diff', 
            'H2H_Avg_Points', 'Home_Home_Form', 'Away_Away_Form']

X = final_df[features]
y = final_df['FTR']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ปรับจูนโมเดล
clf1 = xgb.XGBClassifier(n_estimators=600, learning_rate=0.005, max_depth=5, random_state=42) 
clf2 = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_split=4, random_state=42)
clf3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.8, max_iter=3000))
eclf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3)], voting='soft')

print("\n🧠 กำลังเทรน AI V3.1 (H2H 6 Match Logic)...")
eclf.fit(X_train, y_train)

# =========================================================
# PART 3: ประเมินผล (แก้ไขตรงนี้ให้เริ่มที่ 60%)
# =========================================================
probs = eclf.predict_proba(X_test)
actuals = y_test

print(f"\n{'='*65}")
print(f"🎯 ผลการทดสอบ V3.1 (แสดงผลตั้งแต่ Confidence 60%)")
print(f"{'='*65}")
print(f"{'Confidence':<12} | {'แมตช์ที่เล่น':<12} | {'ทายถูก':<10} | {'ความแม่นยำ (%)':<15}")
print("-" * 65)

# ✅ เพิ่ม 0.60 และ 0.65 เข้าไปใน List นี้
thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

for t in thresholds:
    correct = 0
    total_played = 0
    for i, prob_set in enumerate(probs):
        if np.max(prob_set) >= t:
            total_played += 1
            if np.argmax(prob_set) == actuals[i]:
                correct += 1
    
    if total_played > 0:
        acc = (correct / total_played) * 100
        marker = ""
        if 80 <= acc < 100: marker = "👈 แนะนำ"
        elif acc == 100: marker = "🔥 เทพเจ้า"
        
        print(f"เกณฑ์ > {t*100:.0f}%   | {total_played:<12} | {correct:<10} | {acc:.2f}% {marker}")

joblib.dump(eclf, 'model_v3_specific.pkl')
print(f"\n💾 บันทึกโมเดลสมบูรณ์: 'model_v3_specific.pkl' เรียบร้อย!")