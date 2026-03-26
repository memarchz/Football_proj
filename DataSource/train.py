import pandas as pd
import joblib # เอาไว้เซฟโมเดล
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. โหลดข้อมูลที่คลีนมาแล้ว
print("📂 กำลังโหลดข้อมูล...")
data = pd.read_csv('clean_premier_league.csv')

# 2. เลือกคอลัมน์ที่จะใช้สอน (Features) และคอลัมน์เฉลย (Target)
feature_cols = [
    'AvgH', 'AvgD', 'AvgA',             # ราคาต่อรอง
    'Home_Form', 'Home_Avg_Shots',      # ค่าพลังเจ้าบ้าน
    'Away_Form', 'Away_Avg_Shots',      # ค่าพลังทีมเยือน
    'H2H_Points'                        # สถิติแพ้ทาง
]

X = data[feature_cols]  # ข้อมูลสำหรับวิเคราะห์
y = data['FTR']         # ผลลัพธ์ (H, D, A)

# 3. แบ่งข้อมูล: 80% เอาไว้สอน (Train), 20% เอาไว้สอบวัดผล (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. สร้างและฝึกสอนโมเดล (ใช้ Random Forest)
print("🧠 กำลังฝึกสอน AI (Training)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. วัดผลความแม่นยำ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"✅ เทรนเสร็จสิ้น!")
print(f"🎯 ความแม่นยำของโมเดล (Accuracy): {accuracy * 100:.2f}%")
print("-" * 30)
print("รายงานผลละเอียด:")
print(classification_report(y_test, y_pred))

# 6. บันทึกโมเดลเก็บไว้ใช้งาน (Save)
joblib.dump(model, 'my_football_model.pkl')
print("💾 บันทึกไฟล์โมเดลชื่อ 'my_football_model.pkl' เรียบร้อยแล้ว")