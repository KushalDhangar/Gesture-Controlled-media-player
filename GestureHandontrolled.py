import cv2
import numpy as np
import math
import pyautogui as p

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Color Adjustments")
cv2.createTrackbar("HMin", "Color Adjustments", 0, 179, lambda x: None)
cv2.createTrackbar("HMax", "Color Adjustments", 179, 179, lambda x: None)
cv2.createTrackbar("SMin", "Color Adjustments", 0, 255, lambda x: None)
cv2.createTrackbar("SMax", "Color Adjustments", 255, 255, lambda x: None)
cv2.createTrackbar("VMin", "Color Adjustments", 0, 255, lambda x: None)
cv2.createTrackbar("VMax", "Color Adjustments", 255, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 2)
    frame = cv2.resize(frame, (600, 500))
    cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
    crop_image = frame[1:500, 0:300]

    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HMin", "Color Adjustments")
    h_max = cv2.getTrackbarPos("HMax", "Color Adjustments")
    s_min = cv2.getTrackbarPos("SMin", "Color Adjustments")
    s_max = cv2.getTrackbarPos("SMax", "Color Adjustments")
    v_min = cv2.getTrackbarPos("VMin", "Color Adjustments")
    v_max = cv2.getTrackbarPos("VMax", "Color Adjustments")

    lower_skin = np.array([h_min, s_min, v_min])
    upper_skin = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(crop_image, [max_contour], -1, (0, 255, 0), 2)

        hull = cv2.convexHull(max_contour)
        cv2.drawContours(crop_image, [hull], -1, (0, 0, 255), 2)

        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull_indices)

        count_defects = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                a = distance(start, end)
                b = distance(start, far)
                c = distance(end, far)

                if b != 0 and c != 0:
                    cosine_angle = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
                    angle = math.acos(cosine_angle) * 180 / math.pi

                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(crop_image, far, 8, (0, 0, 255), -1)

        total_fingers = count_defects + 1
        cv2.putText(frame, f'Fingers: {total_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if total_fingers == 1:
            p.press('volumeup')
        elif total_fingers == 2:
            p.press('volumedown')
        elif total_fingers == 3:
            p.press('playpause')
        elif total_fingers == 4:
            p.press('nexttrack')
        elif total_fingers == 5:
            p.press('prevtrack')

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
