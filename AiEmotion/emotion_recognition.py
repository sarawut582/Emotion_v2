import cv2
from deepface import DeepFace
import json
from datetime import datetime
from collections import Counter

# เปิดการจับภาพจากกล้อง (กล้องตัวแรก)
cap = cv2.VideoCapture(0)

# โหลดโมเดล Cascade Classifier สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ตัวแปรสำหรับเก็บข้อมูล
classroom_id = "05-0404"  # รหัสห้องเรียน
session_id = "002"  # คาบเรียน
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # เวลาปัจจุบัน

# ตัวแปรสำหรับนับอารมณ์
emotion_count = {
    'angry_count': 0,
    'disgust_count': 0,
    'fear_count': 0,
    'happy_count': 0,
    'sad_count': 0,
    'surprise_count': 0,
    'neutral_count': 0
}

# ตัวแปรสำหรับความมั่นใจในแต่ละอารมณ์
emotion_confidence = {
    'angry_confidence': 0.0,
    'disgust_confidence': 0.0,
    'fear_confidence': 0.0,
    'happy_confidence': 0.0,
    'sad_confidence': 0.0,
    'surprise_confidence': 0.0,
    'neutral_confidence': 0.0
}

# ตัวแปรเก็บอารมณ์โดดเด่นของแต่ละใบหน้า
dominant_emotions_all = []

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถจับภาพได้จากกล้อง")
        break

    # แปลงภาพเป็นขาวดำสำหรับการตรวจจับใบหน้า
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้าในภาพ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # วาดกรอบรอบใบหน้าที่ตรวจพบ
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # นับจำนวนใบหน้า
    student_count = len(faces)

    # แสดงภาพที่จับได้
    cv2.imshow("Frame", frame)

    # ตัวแปรสำหรับเก็บผลการวิเคราะห์อารมณ์ของแต่ละคน
    dominant_emotions_all.clear()  # เคลียร์อารมณ์จากรอบก่อนหน้า

    # ส่งภาพไปให้ DeepFace วิเคราะห์อารมณ์ทีละใบหน้า
    try:
        for (x, y, w, h) in faces:
            # ครอบภาพใบหน้าจากเฟรม
            face_img = frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

            # เก็บอารมณ์ที่โดดเด่นของแต่ละใบหน้า
            if result and len(result) > 0:
                dominant_emotions_all.append(result[0]['dominant_emotion'])

                # อัปเดตค่าการนับอารมณ์และความมั่นใจ
                emotion_data = result[0]['emotion']
                for emotion, count in emotion_count.items():
                    emotion_name = emotion.split('_')[0]
                    if emotion_name in emotion_data:
                        emotion_count[emotion] += 1
                        emotion_confidence[emotion + '_confidence'] = emotion_data[emotion_name]

    except Exception as e:
        print(f"Error analyzing image: {e}")

    # คำนวณอารมณ์ที่พบและจำนวนของแต่ละอารมณ์
    emotion_counter = Counter(dominant_emotions_all)
    
    # สร้างข้อมูล JSON
    data = {
        "classroom_id": classroom_id,
        "timestamp": timestamp,
        "student_count": student_count,
        "session_id": session_id,
        "dominant_emotions": {emotion: emotion_counter[emotion] for emotion in emotion_counter},  # แสดงจำนวนของแต่ละอารมณ์
        "angry_count": emotion_count['angry_count'],
        "disgust_count": emotion_count['disgust_count'],
        "fear_count": emotion_count['fear_count'],
        "happy_count": emotion_count['happy_count'],
        "sad_count": emotion_count['sad_count'],
        "surprise_count": emotion_count['surprise_count'],
        "neutral_count": emotion_count['neutral_count'],
        "angry_confidence": emotion_confidence['angry_confidence'],
        "disgust_confidence": emotion_confidence['disgust_confidence'],
        "fear_confidence": emotion_confidence['fear_confidence'],
        "happy_confidence": emotion_confidence['happy_confidence'],
        "sad_confidence": emotion_confidence['sad_confidence'],
        "surprise_confidence": emotion_confidence['surprise_confidence'],
        "neutral_confidence": emotion_confidence['neutral_confidence']
    }

    # แสดงผลใน terminal สำหรับ dominant_emotions (เช่น "neutral": 1)
    print(f"Dominant Emotions in the Classroom: {data['dominant_emotions']}")

    # ส่งข้อมูลไปที่ไฟล์ result.json
    with open("result.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    # เมื่อกดปุ่ม 'q' จะออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการจับภาพ
cap.release()
cv2.destroyAllWindows()
