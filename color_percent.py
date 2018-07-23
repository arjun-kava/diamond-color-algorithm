import numpy as np
import cv2

image_path = "data/test/J_54428.jpg"
img = cv2.imread(image_path)

yellow = [255, 255, 0]  # RGB
diff = 50
boundaries = [([yellow[2] - diff, yellow[1] - diff, yellow[0] - diff],
               [yellow[2] + diff, yellow[1] + diff, yellow[0] + diff])]
# in order BGR as opencv represents images as numpy arrays in reverse order

for (lower, upper) in boundaries:
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    ratio_brown = cv2.countNonZero(mask)/(img.size/3)
    print('yellow pixel percentage:', np.round(ratio_brown*100, 2))
    cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    cv2.imshow("images", np.hstack([img, output]))
    cv2.resizeWindow('images', 600, 600)
    cv2.waitKey(0)