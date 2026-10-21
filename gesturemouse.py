import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ==============================
# CONFIG
# ==============================
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.7
SMOOTHING_FACTOR = 0.7
PINCH_THRESHOLD = 35
WINDOW_TITLE = "Gesture Mouse Control"

# ==============================
# 5 GESTURES ONLY:
# 1. 👆 ONE FINGER (Index)     → Move cursor
# 2. ✌️ TWO FINGERS (Index+Middle) → Scroll (move hand up/down)
# 3. 🤏 PINCH (Thumb+Index)     → Left Click
# 4. 🖐️ OPEN HAND (All fingers) → Right Click
# 5. ✊ FIST (All closed)       → Drag & Drop (hand moves while dragging)
# ==============================

# ==============================
# INIT
# ==============================
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_x, prev_y = screen_w // 2, screen_h // 2
last_click_time = 0
prev_scroll_y = 0
dragging = False

# ==============================
# HELPER FUNCTION
# ==============================
def get_fingers_up(landmarks, w, h):
    """Detect which fingers are raised"""
    lm = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks]

    # Check thumb (horizontal)
    thumb_up = lm[4][0] > lm[3][0]  # For right hand

    # Check other fingers (vertical)
    index_up = lm[8][1] < lm[6][1]
    middle_up = lm[12][1] < lm[10][1]
    ring_up = lm[16][1] < lm[14][1]
    pinky_up = lm[20][1] < lm[18][1]

    return [thumb_up, index_up, middle_up, ring_up, pinky_up], lm

def move_mouse_to_finger(index_tip, w, h):
    """Move mouse cursor to finger position"""
    global prev_x, prev_y

    # Map index finger to screen
    target_x = np.interp(index_tip[0], (50, w - 50), (0, screen_w))
    target_y = np.interp(index_tip[1], (50, h - 50), (0, screen_h))

    # Smooth movement
    mouse_x = int(SMOOTHING_FACTOR * target_x + (1 - SMOOTHING_FACTOR) * prev_x)
    mouse_y = int(SMOOTHING_FACTOR * target_y + (1 - SMOOTHING_FACTOR) * prev_y)
    mouse_x = max(0, min(mouse_x, screen_w))
    mouse_y = max(0, min(mouse_y, screen_h))
    prev_x, prev_y = mouse_x, mouse_y

    pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)
    return mouse_x, mouse_y

# ==============================
# MAIN LOOP
# ==============================
print("Gesture Mouse Control Started!")
print("Controls:")
print("  👆 1 finger = Move mouse")
print("  ✌️ 2 fingers = Scroll (move hand up/down)")
print("  🤏 Pinch = Left Click")
print("  🖐️ Open hand = Right Click")
print("  ✊ Fist = Drag & Drop (hand moves while dragging)")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()
    gesture = "None"

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Get finger states
            fingers, lm = get_fingers_up(handLms.landmark, w, h)

            # Get fingertip positions
            thumb_tip = lm[4]
            index_tip = lm[8]
            middle_tip = lm[12]

            # Count fingers (excluding thumb)
            finger_count = sum(fingers[1:])

            # Calculate pinch distance
            pinch_distance = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

            # ==============================
            # GESTURE 1: MOVE CURSOR (1 finger - Index only)
            # ==============================
            if finger_count == 1 and fingers[1] and not fingers[2]:
                gesture = "👆 Moving"
                move_mouse_to_finger(index_tip, w, h)

            # ==============================
            # GESTURE 2: SCROLL (2 fingers - Index + Middle)
            # ==============================
            elif finger_count == 2 and fingers[1] and fingers[2] and not fingers[3]:
                gesture = "✌️ Scrolling"

                # Move cursor while scrolling (optional)
                move_mouse_to_finger(index_tip, w, h)

                # Use vertical movement for scrolling
                if prev_scroll_y != 0:
                    scroll_diff = index_tip[1] - prev_scroll_y
                    if abs(scroll_diff) > 15:
                        scroll_amount = int(scroll_diff / 9)
                        pyautogui.scroll(scroll_amount)
                        direction = "UP" if scroll_amount > 0 else "UP"
                        cv2.putText(frame, f"Scroll {direction}", (40, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                prev_scroll_y = index_tip[1]

            else:
                prev_scroll_y = 0

            # ==============================
            # GESTURE 3: LEFT CLICK (Pinch - Thumb + Index)
            # ==============================
            if pinch_distance < PINCH_THRESHOLD and (current_time - last_click_time) > 0.3:
                if not dragging:
                    gesture = "🤏 Left Click"
                    pyautogui.click()
                    last_click_time = current_time
                    cv2.putText(frame, "LEFT CLICK!", (40, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

            # ==============================
            # GESTURE 4: RIGHT CLICK (Open hand - All fingers up)
            # ==============================
            elif finger_count == 4 and all(fingers[1:]):
                gesture = "🖐️ Right Click"
                if (current_time - last_click_time) > 0.3:
                    pyautogui.rightClick()
                    last_click_time = current_time
                    cv2.putText(frame, "RIGHT CLICK!", (40, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)

            # ==============================
            # GESTURE 5: DRAG & DROP (Fist - All fingers down)
            # ==============================
            elif finger_count == 0:
                gesture = "✊ Dragging"

                if not dragging:
                    # Start dragging
                    pyautogui.mouseDown()
                    dragging = True
                    cv2.putText(frame, "DRAG START", (40, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

                # IMPORTANT: Move mouse while dragging
                move_mouse_to_finger(index_tip, w, h)
                cv2.putText(frame, "DRAGGING... (move hand)", (40, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

            # Release drag if fist is opened
            elif dragging and finger_count > 0:
                pyautogui.mouseUp()
                dragging = False
                cv2.putText(frame, "DROP", (40, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)

            # Display current gesture
            cv2.putText(frame, f"Gesture: {gesture}", (40, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Highlight fingertips
            cv2.circle(frame, index_tip, 8, (0, 255, 0), -1)
            cv2.circle(frame, middle_tip, 8, (0, 255, 0), -1)

    # ==============================
    # TITLE INSIDE THE FRAME
    # ==============================
    cv2.putText(frame, WINDOW_TITLE, (w // 2 - 170, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ==============================
    # INSTRUCTIONS ON SCREEN
    # ==============================
    cv2.putText(frame, "CONTROLS:", (w - 220, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "1 finger = Move", (w - 220, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "2 fingers = Scroll", (w - 220, 95),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Pinch = Left Click", (w - 220, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Open hand = Right Click", (w - 220, 145),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Fist = Drag & Drop", (w - 220, 170),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'q' to quit", (w - 220, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show drag status prominently
    if dragging:
        cv2.putText(frame, "🔴 DRAG MODE ACTIVE - MOVE HAND TO DRAG", (w // 2 - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow(WINDOW_TITLE, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
