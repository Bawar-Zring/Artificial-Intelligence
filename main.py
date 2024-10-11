import numpy as np
import cv2
import mediapipe
import pyautogui
import time

capture_hands = mediapipe.solutions.hands.Hands()
drawing_option = mediapipe.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)
x1 = y1 = x2 = y2 = 0
prev_time = 0
scroll_start_y = 0
mouse_last_x, mouse_last_y = pyautogui.position()

hand_tracking_mode = True

while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if hand_tracking_mode:
        output_hands = capture_hands.process(rgb_image)
        all_Hands = output_hands.multi_hand_landmarks

        if all_Hands:
            for hand in all_Hands:
                hand = all_Hands[0]
                #drawing_option.draw_landmarks(image, hand)
                one_hand_landmarks = hand.landmark

                for id, lm in enumerate(one_hand_landmarks):
                    x = int(lm.x * image_width)
                    y = int(lm.y * image_height)

                    if id == 0:
                        mouse_x = int(screen_width / screen_width * x)
                        mouse_y = int(screen_height / screen_height * y)
                        pyautogui.moveTo(mouse_x * 3, mouse_y * 2)
                        mouse_last_x, mouse_last_y = mouse_x, mouse_y

                    if id == 8:
                        # cv2.circle(image, (x, y), 10, (0, 175, 255), -1)
                        x1 = x
                        y1 = y

                    if id == 12:
                        # cv2.circle(image, (x, y), 10, (0, 175, 255), -1)
                        x3 = x
                        y3 = y

                    if id == 4:
                        # cv2.circle(image, (x, y), 10, (0, 175, 255), -1)
                        x2 = x
                        y2 = y

                dist = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
                dist_right_click = np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))

                if dist < 40:
                    pyautogui.click()

                if dist_right_click < 40:
                    pyautogui.rightClick()

                current_time = time.time()
                if dist < 40:
                    if current_time - prev_time < 0.55:
                        print(current_time - prev_time)
                        pyautogui.doubleClick()
                    prev_time = current_time

                if scroll_start_y == 0:
                    scroll_start_y = y1
                scroll_diff = scroll_start_y - y1

                if  -20 > scroll_diff or scroll_diff > 50:
                    pyautogui.scroll(int(scroll_diff))

    resize_image = cv2.resize(image, (900, 720))

    cv2.imshow('Hand Movement Video', resize_image)

    key = cv2.waitKey(100)

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()