import cv2
import imutils as imutils
import numpy as np

img = cv2.imread("face.jpg")
glasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2. CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 4)
print('Number of detected faces:', len(faces))

for (x, y, w, h) in faces:
    mask_img = np.zeros(img.shape, dtype="uint8")
    cv2.ellipse(mask_img, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (255, 255, 255), -1)
    face_blur = cv2.medianBlur(img, 99)
    img = np.where(mask_img > 0, face_blur, img)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.ellipse(img, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (0, 0, 255), 3)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    eye_x, eye_y, eye_h = eyes[0][0], eyes[0][1], eyes[0][3]
    glasses = imutils.resize(glasses, height=eye_h * 3)
    alpha = glasses[:, :, 3] / 255
    colours = glasses[:, :, :3]
    alpha_mask = np.dstack((alpha, alpha, alpha))
    gh, gw = glasses.shape[:2]
    glass_x = x + w // 6
    glass_y = y + h // 8
    for (ex, ey, ew, eh) in eyes:
        cv2.circle(roi_color, (ex + ew // 2, ey + eh // 2), (ew // 2), (255, 0, 0), 2)

    eyes_back = img[glass_y:glass_y + gh, glass_x:glass_x + gw]
    glasses = eyes_back * (1 - alpha_mask) + colours * alpha_mask
    img[glass_y:glass_y + gh, glass_x:glass_x + gw] = glasses

if __name__ == '__main__':
    cv2.imshow("Eyes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


