import pandas as pd
import joblib
import numpy as np

# 1. โหลดโมเดลและข้อมูล
print("📂 โหลดสมอง AI และฐานข้อมูล...")
model = joblib.load('my_football_model.pkl')
data = pd.read_csv('clean_premier_league.csv')

# =========================================================
# 🛠️ ฟังก์ชันดึงค่าพลังล่าสุดอัตโนมัติ (ไม่ต้องนับมือ)
# =========================================================
def get_latest_team_stats(team_name, side='Home'):
    # หาประวัติทั้งหมดของทีมนี้
    if side == 'Home':
        matches = data[data['HomeTeam'] == team_name]
    else:
        matches = data[data['AwayTeam'] == team_name]
        
    if len(matches) == 0:
        print(f"❌ ไม่พบทีม {team_name} ในฐานข้อมูล")
        return None, None

    # เอาแมตช์ล่าสุดที่มีในข้อมูล
    last_match = matches.iloc[-1]
    
    # ดึงค่า Form และ Avg_Shots ที่คำนวณไว้ล่าสุด
    if side == 'Home':
        return last_match['Home_Form'], last_match['Home_Avg_Shots']
    else:
        return last_match['Away_Form'], last_match['Away_Avg_Shots']

def get_h2h_stats(home, away):
    # ดึงค่า H2H ล่าสุดจากคู่ที่เคยเจอกัน
    matches = data[((data['HomeTeam']==home) & (data['AwayTeam']==away)) | 
                   ((data['HomeTeam']==away) & (data['AwayTeam']==home))]
    
    if len(matches) > 0:
        return matches.iloc[-1]['H2H_Points']
    else:
        return 0 # ไม่เคยเจอกัน

# =========================================================
# 🎮 เริ่มทำนายแมตช์: ARSENAL vs CRYSTAL PALACE
# =========================================================

# 1. ระบุชื่อทีม (ต้องสะกดให้ตรงกับในไฟล์ CSV)
home_team = 'Arsenal'
away_team = 'Crystal Palace'

# 2. ใส่ราคาต่อรองปัจจุบัน (ต้องดูจากเว็บพนันจริง)
# สมมติราคา: Arsenal ต่อหนัก (1.30), เสมอ (5.50), Palace ชนะ (10.00)
current_odds_H = 1.45  
current_odds_D = 4.33
current_odds_A = 7.00

print(f"\n🔍 กำลังวิเคราะห์คู่: {home_team} vs {away_team}")

# ดึงสถิติล่าสุดจากระบบ
h_form, h_shots = get_latest_team_stats(home_team, 'Home')
a_form, a_shots = get_latest_team_stats(away_team, 'Away')
h2h_pts = get_h2h_stats(home_team, away_team)

print(f"📊 สถิติที่ AI เห็น:")
print(f" - {home_team} (ในบ้าน): ฟอร์ม {h_form}/18, โอกาสยิงเฉลี่ย {h_shots:.1f}")
print(f" - {away_team} (เยือน): ฟอร์ม {a_form}/18, โอกาสยิงเฉลี่ย {a_shots:.1f}")
print(f" - H2H Point (อาร์เซนอลข่มแค่ไหน): {h2h_pts}/18")

# เตรียมข้อมูลเข้า AI
input_data = pd.DataFrame([{
    'AvgH': current_odds_H,
    'AvgD': current_odds_D,
    'AvgA': current_odds_A,
    'Home_Form': h_form,
    'Home_Avg_Shots': h_shots,
    'Away_Form': a_form,
    'Away_Avg_Shots': a_shots,
    'H2H_Points': h2h_pts
}])

# ให้ AI ทำนาย
prediction = model.predict(input_data)[0]
probs = model.predict_proba(input_data)[0]
classes = model.classes_ # ดูลำดับ [Away, Draw, Home]

print("-" * 40)
print(f"🧠 AI ฟันธง: {prediction} ", end="")
if prediction == 'H': print(f"(เจ้าบ้าน {home_team} ชนะ!)")
elif prediction == 'A': print(f"(ทีมเยือน {away_team} บุกชนะ!)")
else: print("(เสมอกัน!)")

print(f"\n📈 ความมั่นใจของ AI:")
# Mapping ลำดับให้ถูกต้อง
score_map = {classes[i]: probs[i] for i in range(len(classes))}
print(f" - เจ้าบ้านชนะ: {score_map.get('H', 0)*100:.2f}%")
print(f" - เสมอ:       {score_map.get('D', 0)*100:.2f}%")
print(f" - ทีมเยือนชนะ: {score_map.get('A', 0)*100:.2f}%")
print("-" * 40)