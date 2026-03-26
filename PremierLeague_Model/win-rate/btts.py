import pandas as pd
import numpy as np
import requests
from io import StringIO
import xgboost as xgb
import joblib

# =========================================================
# 1. โหลดข้อมูล + ราคาตลาด
# =========================================================
print("🚀 เริ่มระบบ V6.1 AUDITED BTTS (เพิ่มตารางเช็คความแม่น)...")

urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
]

dfs = []
for url in urls:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.content.decode('latin-1')))
            df.columns = df.columns.str.strip()
            
            # เลือก Column (ต้องมีราคา Goal เพื่อใช้คำนวณ)
            target_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 
                           'B365>2.5', 'B365<2.5']
            
            valid_cols = [c for c in target_cols if c in df.columns]
            if len(valid_cols) > 5:
                if 'B365>2.5' not in df.columns: df['B365>2.5'] = 1.90
                if 'B365<2.5' not in df.columns: df['B365<2.5'] = 1.90
                
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                dfs.append(df[valid_cols])
    except: pass

full_data = pd.concat(dfs, ignore_index=True).sort_values('Date').dropna().reset_index(drop=True)
full_data['BTTS_Target'] = np.where((full_data['FTHG'] > 0) & (full_data['FTAG'] > 0), 1, 0)

# =========================================================
# 2. Feature Engineering
# =========================================================
print("⏳ คำนวณสถิติย้อนหลังและแบ่งชุดทดสอบ...")

def get_btts_market_features(row, full_data):
    # Market Features
    imp_over = (1 / row['B365>2.5']) * 100
    imp_under = (1 / row['B365<2.5']) * 100
    
    past = full_data[full_data['Date'] < row['Date']]
    if len(past) < 50: return pd.Series([0]*7)
    
    h_games = past[(past['HomeTeam'] == row['HomeTeam']) | (past['AwayTeam'] == row['HomeTeam'])].tail(8)
    a_games = past[(past['HomeTeam'] == row['AwayTeam']) | (past['AwayTeam'] == row['AwayTeam'])].tail(8)
    
    if len(h_games) < 5 or len(a_games) < 5: return pd.Series([0]*7)
    
    def get_stats(games, team):
        btts_c = 0
        sco, con = 0, 0
        for _, m in games.iterrows():
            gf = m['FTHG'] if m['HomeTeam'] == team else m['FTAG']
            ga = m['FTAG'] if m['HomeTeam'] == team else m['FTHG']
            if gf > 0 and ga > 0: btts_c += 1
            sco += gf; con += ga
        return btts_c/len(games), sco/len(games), con/len(games)

    h_btts, h_sco, h_con = get_stats(h_games, row['HomeTeam'])
    a_btts, a_sco, a_con = get_stats(a_games, row['AwayTeam'])
    
    pot_score_h = (h_sco + a_con) / 2
    pot_score_a = (a_sco + h_con) / 2
    avg_btts_freq = (h_btts + a_btts) / 2
    
    return pd.Series([imp_over, imp_under, imp_over - imp_under, pot_score_h, pot_score_a, avg_btts_freq, (h_con+a_con)])

cols_new = ['Mkt_Imp_Over', 'Mkt_Imp_Under', 'Mkt_Bias', 'Pot_H', 'Pot_A', 'Freq_BTTS', 'Def_Leak']
full_data[cols_new] = full_data.apply(lambda x: get_btts_market_features(x, full_data), axis=1)

final_df = full_data[full_data['Mkt_Imp_Over'] > 0].dropna()

# =========================================================
# 3. Training & AUDIT (ส่วนสำคัญที่เพิ่มมา)
# =========================================================
features = cols_new
X = final_df[features]
y = final_df['BTTS_Target']

# แบ่งข้อมูล: 80% แรกใช้เทรน, 20% หลังใช้ตรวจสอบความแม่นยำ (Backtest)
train_size = int(len(X) * 0.80)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.015, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# --- สร้างตารางตรวจสอบความแม่นยำ (Accuracy Audit) ---
probs_test = clf.predict_proba(X_test)[:, 1] # เอาเฉพาะ % BTTS Yes
y_test_arr = y_test.values

print(f"\n{'='*90}")
print(f"📊 ตารางตรวจสอบความแม่นยำ (BACKTEST REPORT)")
print(f"ทดสอบกับ {len(X_test)} แมตช์ล่าสุด เพื่อดูว่า AI พูดจริงหรือมั่ว")
print(f"{'-'*90}")
print(f"{'AI Confidence':<15} | {'แมตช์ที่พบ':<10} | {'BTTS เข้าจริง':<12} | {'ไม่เข้า':<8} | {'แม่นยำจริง %':<12} | {'เกรด'}")
print(f"{'-'*90}")

# แบ่งช่วงความมั่นใจ
bins = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 1.00]
labels = ["40-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70%+"]

for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    # หาแมตช์ที่ AI มั่นใจในช่วงนี้
    mask = (probs_test >= low) & (probs_test < high)
    total = np.sum(mask)
    
    if total > 0:
        hits = np.sum(y_test_arr[mask] == 1) # BTTS เกิดขึ้นจริง
        missed = total - hits
        real_acc = (hits / total) * 100
        
        # ตัดเกรด
        grade = ""
        if real_acc >= 65: grade = "🔥 GOD TIER"
        elif real_acc >= 55: grade = "✅ PROFIT"
        elif real_acc >= 50: grade = "🤔 FAIR"
        else: grade = "💀 RISKY"
        
        print(f"{low*100:.0f}% - {high*100:.0f}%     | {total:<10} | {hits:<12} | {missed:<8} | {real_acc:.1f}%       | {grade}")

print(f"{'='*90}")

# เทรนซ้ำด้วยข้อมูลทั้งหมดเพื่อเตรียมใช้งานจริง
clf.fit(X, y) 

# =========================================================
# 4. PREDICTION ZONE (เหมือนเดิม)
# =========================================================
print("\n🔮 กำลังวิเคราะห์คู่บอลวันนี้...")

# --- ใส่ข้อมูลจริงที่นี่ ---
match_input = [
    {"Home": "Man United", "Away": "Liverpool",  "Odds_BTTS": [1.44, 2.62], "Odds_O2.5": 1.50},
    {"Home": "Ipswich",    "Away": "Man City",   "Odds_BTTS": [1.80, 1.95], "Odds_O2.5": 1.40},
    {"Home": "Everton",    "Away": "Wolves",     "Odds_BTTS": [1.95, 1.80], "Odds_O2.5": 2.10}
]

print(f"{'MATCH':<25} | {'AI %':<6} | {'WEB %':<6} | {'DIFF':<6} | {'PREDICTION'}")
print(f"{'-'*80}")

for m in match_input:
    home, away = m['Home'], m['Away']
    odds_yes = m['Odds_BTTS'][0]
    odds_over = m['Odds_O2.5']
    
    try:
        last_stats = final_df[(final_df['HomeTeam']==home) | (final_df['AwayTeam']==home)].iloc[-1]
        
        imp_o = (1/odds_over)*100
        imp_u = (1/(3.8 - odds_over))*100 
        
        input_vec = pd.DataFrame([[
            imp_o, imp_u, imp_o - imp_u,
            last_stats['Pot_H'], last_stats['Pot_A'], 
            last_stats['Freq_BTTS'], last_stats['Def_Leak']
        ]], columns=features)
        
        prob_yes = clf.predict_proba(input_vec)[0][1] * 100
        market_prob = (1 / odds_yes) * 100
        diff = prob_yes - market_prob
        
        rec = "Skip"
        if prob_yes >= 60: rec = "YES"
        elif prob_yes <= 40: rec = "NO"
        
        # ใส่ Tag Value Bet
        value_tag = ""
        if diff > 5 and prob_yes > 55: value_tag = "💎 (Value)"
        
        print(f"{home:<12} vs {away:<10} | {prob_yes:.1f}% | {market_prob:.1f}% | {diff:+.1f}% | {rec} {value_tag}")
        
    except: print(f"{home} vs {away}: ไม่พบข้อมูล")
print(f"{'='*80}")
# ... (ส่วนบนเหมือนเดิม) ...

# =========================================================
# 5. 🔮 PREDICTION WITH CALIBRATION (จูนค่าตามความจริง)
# =========================================================
print(f"\n{'='*95}")
print("   🔮 V7.0 CALIBRATED PREDICTOR (ปรับแก้ตามผล Audit)")
print("   💡 หลักการ: สวน AI ในจุดที่มันมั่นใจเกินเหตุ & ตามในจุดที่มันมองข้าม")
print(f"{'='*95}")

match_input = [
    # ลองใส่คู่ที่คุณสนใจ (ตัวอย่าง)
    {"Home": "Man United", "Away": "Liverpool",  "Odds_BTTS": [1.44, 2.62], "Odds_O2.5": 1.50},
    {"Home": "Ipswich",    "Away": "Man City",   "Odds_BTTS": [1.80, 1.95], "Odds_O2.5": 1.40},
    {"Home": "Everton",    "Away": "Wolves",     "Odds_BTTS": [1.95, 1.80], "Odds_O2.5": 2.10},
    {"Home": "Fulham",     "Away": "Brentford",  "Odds_BTTS": [1.60, 2.20], "Odds_O2.5": 1.70} 
]

print(f"{'MATCH':<25} | {'Raw AI':<8} | {'Real Prob':<10} | {'Web Prob':<8} | {'Diff':<6} | {'คำแนะนำ'}")
print(f"{'-'*95}")

for m in match_input:
    home, away = m['Home'], m['Away']
    odds_yes = m['Odds_BTTS'][0]
    odds_over = m['Odds_O2.5']
    
    try:
        last_stats = final_df[(final_df['HomeTeam']==home) | (final_df['AwayTeam']==home)].iloc[-1]
        
        # Prepare Feature
        imp_o = (1/odds_over)*100
        imp_u = (1/(3.8 - odds_over))*100 
        input_vec = pd.DataFrame([[
            imp_o, imp_u, imp_o - imp_u,
            last_stats['Pot_H'], last_stats['Pot_A'], 
            last_stats['Freq_BTTS'], last_stats['Def_Leak']
        ]], columns=features)
        
        # 1. ได้ค่า Raw Confidence จาก AI
        raw_ai = clf.predict_proba(input_vec)[0][1] * 100
        
        # 2. 🔥 CALIBRATION STEP (แปลงค่าตามตาราง Audit ของคุณ) 🔥
        # นี่คือหัวใจสำคัญ: แปลงสิ่งที่ AI คิด เป็น สิ่งที่เกิดขึ้นจริง
        real_prob = 0
        note = ""
        
        if raw_ai < 50: 
            real_prob = 61.4 # AI มองต่ำ แต่ของจริงเข้าบ่อย (Golden Zone)
            note = "💰 Hidden Gem"
        elif 50 <= raw_ai < 55:
            real_prob = 59.3 # ยังน่าเล่น
            note = "✅ Good"
        elif 55 <= raw_ai < 60:
            real_prob = 53.3 # เริ่มเสี่ยง
            note = "🤔 Fair"
        elif 60 <= raw_ai < 65:
            real_prob = 57.1 # กลับมาดีนิดหน่อย
            note = "✅ Okay"
        else: # 65% ขึ้นไป (Zone อันตราย)
            real_prob = 47.0 # AI มั่นใจมาก แต่ของจริงวูบ
            note = "💀 TRAP!"

        # 3. เทียบกับราคาเว็บ
        market_prob = (1 / odds_yes) * 100
        diff = real_prob - market_prob # ใช้ Real Prob มาคำนวณความคุ้มค่าแทน
        
        # ตัดเกรดคำแนะนำ
        action = ""
        if diff > 5: action = f"BET YES ({note})"
        elif diff < -5: action = "BET NO (Suan)"
        else: action = "PASS"
        
        if "TRAP" in note: action = "AVOID / BET NO"

        print(f"{home:<12} vs {away:<10} | {raw_ai:.1f}%   | {real_prob:.1f}%     | {market_prob:.1f}%   | {diff:+.1f}% | {action}")
        
    except: print(f"{home} vs {away}: Data not found")

print(f"{'='*95}")
print("หมายเหตุ: 'Real Prob' คือค่าที่ปรับจูนจากตาราง Audit ของคุณแล้ว (เชื่อถือได้มากกว่า Raw AI)")