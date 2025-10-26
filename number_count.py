import cv2
import numpy as np

cap = cv2.VideoCapture(0)

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

    # Noise removal
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)
    _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

            hull_indices = cv2.convexHull(hand, returnPoints=False)
    
            if len(hull_indices) > 3:
                defects = cv2.convexityDefects(hand, hull_indices)
                
                if defects is not None:
                    finger_count = 0
                    
                    for i in range(defects.shape[0]):
                        starting_point, end_point_index, farthest_point_index, defect_depth = defects[i, 0]
                        start = tuple(hand[starting_point][0])
                        end = tuple(hand[end_point_index][0])
                        far = tuple(hand[farthest_point_index][0])
                        
                        # Calculate triangle sides
                        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        
                        # Calculate angle
                        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                        
                        # Count valid defects
                        if angle <= np.pi/2 and defect_depth > 10000:
                            finger_count += 1
                            cv2.circle(frame, far, 5, (0, 0, 255), -1)
                    
                    finger_count = min(finger_count + 1, 5)
                    
                    cv2.putText(frame, f"Fingers: {finger_count}", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Clean Mask', clean_mask)
    cv2.imshow('Finger Counter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()