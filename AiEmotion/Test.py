import cv2
from deepface import DeepFace
import json
from datetime import datetime
from collections import Counter

# IP ของกล้อง Tapo C320WS
camera_ip = 'rtsp://Face408:cpe408@172.20.10.4:554/stream1'

# ตั้งค่ากล้อง RTSP โดยใช้ FFmpeg หรือ GStreamer
cap = cv2.VideoCapture(camera_ip, cv2.CAP_FFMPEG)

# ตรวจสอบการเชื่อมต่อกล้อง RTSP
if not cap.isOpened():
    print("Error: ไม่สามารถเชื่อมต่อกับกล้อง RTSP โดยใช้ FFmpeg")
    print("กำลังลองเชื่อมต่อโดยใช้ GStreamer...")
    cap = cv2.VideoCapture(f"rtspsrc location={camera_ip} latency=0 ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: ไม่สามารถเชื่อมต่อกับกล้อง RTSP โดยใช้ GStreamer")
        exit()

print("✅ เชื่อมต่อกับกล้อง RTSP สำเร็จ!")

# ตั้งค่าความละเอียดที่เหมาะสม
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# โหลดโมเดล Cascade Classifier สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ตัวแปรสำหรับเก็บข้อมูล
classroom_id = "05-0404"
session_id = "002"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

dominant_emotions_all = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถจับภาพจากกล้อง RTSP ได้")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    student_count = len(faces)
    dominant_emotions_all.clear()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = frame[y:y + h, x:x + w]
        
        try:
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            if result and len(result) > 0:
                dominant_emotions_all.append(result[0]['dominant_emotion'])
                
                emotion_data = result[0]['emotion']
                for emotion, count in emotion_count.items():
                    emotion_name = emotion.split('_')[0]
                    if emotion_name in emotion_data:
                        emotion_count[emotion] += 1
                        emotion_confidence[emotion + '_confidence'] = emotion_data[emotion_name]
        except Exception as e:
            print(f"Error analyzing image: {e}")
    
    emotion_counter = Counter(dominant_emotions_all)
    data = {
        "classroom_id": classroom_id,
        "timestamp": timestamp,
        "student_count": student_count,
        "session_id": session_id,
        "dominant_emotions": {emotion: emotion_counter[emotion] for emotion in emotion_counter},
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
    
    print(f"Dominant Emotions in the Classroom: {data['dominant_emotions']}")
    
    with open("result.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
