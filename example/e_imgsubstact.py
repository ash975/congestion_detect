import cv2
import numpy as np

# src img
img1 = cv2.imread('../../assets/frame/frame186.png', 0)
img2 = cv2.imread('../../assets/background/background1800.png', 0)

# mask
segements1 = cv2.imread('../../assets/segements.png', 0)
# mask 二值化
ret1, segements = cv2.threshold(segements1, 127, 255, cv2.THRESH_BINARY)

# mask src
img1 = cv2.bitwise_and(img1, img1, mask=segements)
img2 = cv2.bitwise_and(img2, img2, mask=segements)

# 比较差异
dst = img1 - img2

# 二值化 dst
ret1, dst2 = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)

# 对二值化 dst 进行处理填充 去噪声
closed = cv2.erode(dst2, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=10)

# 显示图片
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
cv2.imshow('org', dst)
cv2.imshow('dst', dst2)
cv2.imshow('proccessed', closed)

c_segements = np.sum(segements)
c_dst2 = np.sum(dst2)


print(c_segements, c_dst2)
print('percent', c_dst2/c_segements*1.00)

# 终止
k = cv2.waitKey(0)
cv2.destroyAllWindows()
