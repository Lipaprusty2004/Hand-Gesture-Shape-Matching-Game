import cv2
import mediapipe as mp
import numpy as np
import math

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

# Colors for shapes
colors = {
    "circle": (0, 255, 255),
    "square": (255, 0, 0),
    "triangle": (255, 0, 255)
}

# Target big shapes
targets = [
    {"x": 150, "y": 100, "shape": "circle", "size": 60},
    {"x": 350, "y": 100, "shape": "square", "size": 80},
    {"x": 550, "y": 100, "shape": "triangle", "size": 90}
]

# Small draggable shapes
draggables = [
    {"x": 100, "y": 400, "shape": "circle", "size": 30, "dragging": False},
    {"x": 300, "y": 400, "shape": "square", "size": 40, "dragging": False},
    {"x": 500, "y": 400, "shape": "triangle", "size": 45, "dragging": False}
]

# Draw shape function
def draw_shape(img, shape, x, y, size, color, thickness=2, fill=False):
    if shape == "circle":
        cv2.circle(img, (x, y), size, color, -1 if fill else thickness)
    elif shape == "square":
        cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1 if fill else thickness)
    elif shape == "triangle":
        pts = np.array([
            [x, y - size],
            [x - size, y + size],
            [x + size, y + size]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        if fill:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

# Distance function
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Main loop
cap = cv2.VideoCapture(0)
match_message = ""
match_color = (0, 255, 0)
message_timer = 0
dragged_shape = None

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw target shapes
    for t in targets:
        draw_shape(frame, t["shape"], t["x"], t["y"], t["size"], colors[t["shape"]], thickness=5)

    # Draw draggable shapes
    for shape in draggables:
        draw_shape(frame, shape["shape"], shape["x"], shape["y"], shape["size"], colors[shape["shape"]], fill=True)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index and thumb coordinates
            index = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]

            index_pos = (int(index.x * w), int(index.y * h))
            thumb_pos = (int(thumb.x * w), int(thumb.y * h))
            pinch_point = ((index_pos[0] + thumb_pos[0]) // 2, (index_pos[1] + thumb_pos[1]) // 2)
            pinch_distance = distance(index_pos, thumb_pos)

            # Draw pinch point
            cv2.circle(frame, index_pos, 8, (255, 255, 255), -1)
            cv2.circle(frame, thumb_pos, 8, (255, 255, 255), -1)
            cv2.circle(frame, pinch_point, 10, (255, 255, 0), -1)

            is_pinch = pinch_distance < 40

            if is_pinch:
                if dragged_shape is None:
                    for shape in draggables:
                        sx, sy = shape["x"], shape["y"]
                        if distance(pinch_point, (sx, sy)) < shape["size"] + 10:
                            shape["dragging"] = True
                            dragged_shape = shape
                            break
                else:
                    dragged_shape["x"], dragged_shape["y"] = pinch_point
            else:
                if dragged_shape:
                    for target in targets:
                        tx, ty = target["x"], target["y"]
                        if distance((dragged_shape["x"], dragged_shape["y"]), (tx, ty)) < target["size"]:
                            if dragged_shape["shape"] == target["shape"]:
                                match_message = "MATCH"
                                match_color = (0, 255, 0)  # Green
                            else:
                                match_message = "NOT MATCH"
                                match_color = (0, 0, 255)  # Red
                            message_timer = 30
                    dragged_shape["dragging"] = False
                    dragged_shape = None

    # Show match/not match text
    if message_timer > 0:
        cv2.putText(frame, match_message, (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, match_color, 4)
        message_timer -= 1

    cv2.imshow("Gesture Shape Matcher", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

