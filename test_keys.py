import cv2

cv2.namedWindow("preview")

while True:
    key = cv2.waitKey(0)
    if key == 27: # exit on ESC
        break

    print(key)

cv2.destroyAllWindows()