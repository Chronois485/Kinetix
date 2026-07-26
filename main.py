import time
import math
import os
import urllib.request
import cv2
import mediapipe as mp
import wayland_automation as wa

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

SCREEN_WIDTH = 1920  
SCREEN_HEIGHT = 1080

prev_x, prev_y = 0, 0
SMOOTHING = 5
SENSITITIVITY = 2.5

if not os.path.exists(MODEL_PATH): 
    print("No model found.\nDownloading...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
        )

HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Великий палець
        (0, 5), (5, 6), (6, 7), (7, 8), # Вказівний
        (5, 9), (9, 10), (10, 11), (11, 12), # Середній
        (9, 13), (13, 14), (14, 15), (15, 16), # Безіменний
        (13, 17), (17, 18), (18, 19), (19, 20), # Мізінець
        (0, 17), (5, 9), (9, 13) # Решта руки
        ]

cap = cv2.VideoCapture(0)

pinch_start_x, pinch_start_y = 0, 0
mouse_start_x, mouse_start_y = 0, 0
pinch_toggle = False
click_toggle = False
right_click_toggle = False
curr_x = 0
curr_y = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        succsess, frame = cap.read()
        if not succsess:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                points = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                x_thumb, y_thumb = points[4]
                x_index, y_index = points[8]
                x_middle, y_middle = points[12]
                x_ring, y_ring = points[16]

                distance_thumd_index = math.hypot(x_index - x_thumb, y_index - y_thumb)
                distance_thumd_middle = math.hypot(x_middle - x_thumb, y_middle - y_thumb)
                distance_thumd_ring = math.hypot(x_ring - x_thumb, y_ring - y_thumb)


                cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (0, 0, 255), 2)
                cv2.line(frame, (x_thumb, y_thumb), (x_middle, y_middle), (255, 0, 0), 2)
                cv2.line(frame, (x_thumb, y_thumb), (x_ring, y_ring), (255, 0, 255), 2)


                cxr, cyr = (x_thumb + x_ring) // 2, (y_thumb + y_ring) // 2
                cv2.circle(frame, (cxr, cyr), 6, (255, 0, 255), -1)

                cxm, cym = (x_thumb + x_middle) // 2, (y_thumb + y_middle) // 2
                cv2.circle(frame, (cxm, cym), 6, (255, 0, 0), -1)

                cx, cy = (x_thumb + x_index) // 2, (y_thumb + y_index) // 2
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                
                if distance_thumd_index < 30:
                    if not pinch_toggle:
                        pinch_start_x, pinch_start_y = cx, cy
                        mouse_start_x, mouse_start_y = prev_x, prev_y
                        pinch_toggle = True

                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
    
                                    
                    dx = int((cx - pinch_start_x) * (SCREEN_WIDTH / w) * SENSITITIVITY)
                    dy = int((cy - pinch_start_y) * (SCREEN_HEIGHT / h) * SENSITITIVITY)
                    target_x = max(0, min(int(mouse_start_x + dx), SCREEN_WIDTH - 1))
                    target_y = max(0, min(int(mouse_start_y + dy), SCREEN_HEIGHT - 1))
                    cv2.putText(frame, f"Pinch dx:{dx} dy:{dy}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                    curr_x = prev_x + (target_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (target_y - prev_y) / SMOOTHING
                    print(f"x: {int(curr_x)}, y: {int(curr_y)}")

                    wa.click(int(curr_x), int(curr_y), "nothing")
                    prev_x, prev_y = curr_x, curr_y
                elif distance_thumd_middle < 30:
                    if not click_toggle:
                        click_toggle = True
                        wa.click(int(curr_x), int(curr_y), "left")
                    cv2.circle(frame, (cxm, cym), 10, (0, 255, 0), -1)
                elif distance_thumd_ring < 30:
                    if not right_click_toggle:
                        right_click_toggle = True
                        wa.click(int(curr_x), int(curr_y), "right")
                    cv2.circle(frame, (cxr, cyr), 10, (0, 255, 0), -1)
                else:
                    pinch_toggle = False
                    click_toggle = False
                    right_click_toggle = False

                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 1)
        cv2.imshow("Kinetix", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
