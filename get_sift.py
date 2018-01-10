import cv2
import numpy as np

img_name='test3.jpg'
img = cv2.imread(img_name)
cv2.imwrite('original.jpg',img)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray.jpg',gray)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
sift_info=np.zeros([kp.__len__(), 5])
for i in range(kp.__len__()):
    k=kp[i]
    sift_info[i, 0] = k.angle
    sift_info[i, 1] = k.octave
    sift_info[i, 2] = k.pt[0]
    sift_info[i, 3] = k.pt[1]
    sift_info[i, 4] = k.size

img=cv2.drawKeypoints(gray,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)



