import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Range of possible skin colour in YCrCb
lower_skin = np.array([0, 133, 77])
upper_skin = np.array([255, 173, 127])

# Calibration phase
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500, 
    dist2Threshold=800,
    detectShadows=False
)

frame_count = 0
calibration_frames = 200

while frame_count < calibration_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    bg_subtractor.apply(frame, learningRate=0.5)

    cv2.putText(frame, f"Calibrating: {calibration_frames - frame_count}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Calibration', frame)
    cv2.waitKey(1)
    frame_count += 1
cv2.destroyWindow('Calibration')

# Detection phase
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame, learningRate=0.00001)

    skin_mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb), lower_skin, upper_skin)

    combined_mask = cv2.bitwise_and(fg_mask, skin_mask)
    
    # Noise removal
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find only LARGE contours
    if contours:
        large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
        
        if large_contours:
            hand = max(large_contours, key=cv2.contourArea)
            # Draw convex hull for smoother shape
            hull = cv2.convexHull(hand)

            overlay = frame.copy()
            cv2.drawContours(overlay, [hull], -1, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)
    
    cv2.imshow('Clean Mask', combined_mask)
    cv2.imshow('Skin Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()