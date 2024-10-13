import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import time

# Setup MediaPipe Hand Solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Open camera
camera = cv2.VideoCapture(0)

# Smoothing parameters for cursor movement
prev_mouse_x, prev_mouse_y = 0, 0
alpha = 0.5  # Smoothing factor

# Timing for gestures
last_click_time = 0
last_scroll_time = time.time()


# Function to smooth cursor movement
def move_cursor(x, y, width, height):
    global prev_mouse_x, prev_mouse_y
    mouse_x = int(x * screen_width / width)
    mouse_y = int(y * screen_height / height)

    # Apply smoothing
    smoothed_mouse_x = int(alpha * mouse_x + (1 - alpha) * prev_mouse_x)
    smoothed_mouse_y = int(alpha * mouse_y + (1 - alpha) * prev_mouse_y)

    pyautogui.moveTo(smoothed_mouse_x, smoothed_mouse_y)
    prev_mouse_x, prev_mouse_y = smoothed_mouse_x, smoothed_mouse_y


# Function to detect gestures for scrolling
def detect_scroll(dist_index_middle):
    global last_scroll_time
    current_time = time.time()

    # Scroll Up
    if dist_index_middle < 35 and current_time - last_scroll_time > 0.5:
        pyautogui.scroll(150)
        last_scroll_time = current_time

    # Scroll Down
    elif dist_index_middle > 85 and current_time - last_scroll_time > 0.5:
        pyautogui.scroll(-150)
        last_scroll_time = current_time


# Function to detect clicks
def detect_clicks(dist_thumb_index, dist_thumb_middle):
    global last_click_time
    current_time = time.time()

    # Single Click
    if dist_thumb_index < 40:
        pyautogui.click()

    # Double Click
    if dist_thumb_index < 30 and current_time - last_click_time < 0.75:
        pyautogui.doubleClick()
        last_click_time = current_time

    # Right Click
    if dist_thumb_middle < 40:
        pyautogui.rightClick()


# Main loop
while True:
    success, image = camera.read()
    if not success:
        break

    # Get image dimensions and flip the image horizontally
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape

    # Convert the image to RGB for hand detection
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_image)
    all_hands = output.multi_hand_landmarks

    # Process hand landmarks if detected
    if all_hands:
        for hand_landmarks in all_hands:
            # Get coordinates for index, middle, and thumb fingers
            x_index = y_index = x_middle = y_middle = x_thumb = y_thumb = 0

            for idx, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * image_width), int(lm.y * image_height)

                # Move the cursor based on wrist coordinates (landmark 0)
                if idx == 0:
                    move_cursor(x, y, image_width, image_height)

                if idx == 8:  # Index finger
                    x_index, y_index = x, y
                if idx == 12:  # Middle finger
                    x_middle, y_middle = x, y
                if idx == 4:  # Thumb
                    x_thumb, y_thumb = x, y

            # Calculate distances between key points
            dist_index_middle = np.linalg.norm(np.array([x_middle, y_middle]) - np.array([x_index, y_index]))
            dist_thumb_index = np.linalg.norm(np.array([x_index, y_index]) - np.array([x_thumb, y_thumb]))
            dist_thumb_middle = np.linalg.norm(np.array([x_middle, y_middle]) - np.array([x_thumb, y_thumb]))

            # Detect scrolling
            detect_scroll(dist_index_middle)

            # Detect clicks
            detect_clicks(dist_thumb_index, dist_thumb_middle)

    # Resize and display the image
    resized_image = cv2.resize(image, (900, 720))
    cv2.imshow('Hand Movement Video', resized_image)

    # Break the loop if 'q' is pressed or window is closed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Hand Movement Video', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
