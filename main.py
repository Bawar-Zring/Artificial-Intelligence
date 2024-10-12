import numpy as np
import cv2
import mediapipe
import pyautogui
import time

from scipy.optimize import brent

capture_hands = mediapipe.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)

last_scroll_time = time.time()
prev_time = 0

while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_hands = capture_hands.process(rgb_image)
    all_Hands = output_hands.multi_hand_landmarks

    if all_Hands:
        for hand in all_Hands:
            hand = all_Hands[0]
            one_hand_landmarks = hand.landmark

            x_index = y_index = x_middle = y_middle = x_thumb = y_thumb = 0

            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                if id == 0:
                    mouse_x = int(screen_width / screen_width * x)
                    mouse_y = int(screen_height / screen_height * y)
                    pyautogui.moveTo(mouse_x * 3, mouse_y * 3)

                if id == 8:  # index finger
                    x_index, y_index = x, y
                if id == 12:  # middle finger
                    x_middle, y_middle = x, y
                if id == 4:  # thumb
                    x_thumb, y_thumb = x, y

            dist_index_middle = np.linalg.norm(np.array([x_middle, y_middle]) - np.array([x_index, y_index]))
            dist_thumb_index = np.linalg.norm(np.array([x_index, y_index]) - np.array([x_thumb, y_thumb]))
            dist_thumb_middle = np.linalg.norm(np.array([x_middle, y_middle]) - np.array([x_thumb, y_thumb]))

            current_time = time.time()

            if dist_index_middle < 35:
                pyautogui.scroll(100)

            elif dist_index_middle > 85:
                pyautogui.scroll(-100)

            if dist_thumb_index < 40:
                pyautogui.click()

            if dist_thumb_index < 30:
                if current_time - prev_time < 0.75:
                    pyautogui.doubleClick()
                prev_time = current_time

            if dist_thumb_middle < 40:
                pyautogui.rightClick()

    resize_image = cv2.resize(image, (900, 720))
    cv2.imshow('Hand Movement Video', resize_image)

    key = cv2.waitKey(100)
    if key == ord('q'):
        break

    if cv2.getWindowProperty('Hand Movement Video', cv2.WND_PROP_VISIBLE) < 1:
        close_window = True
        break

camera.release()
cv2.destroyAllWindows()