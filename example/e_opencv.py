import cv2

# 载入图像
im = cv2.imread('../../asserts/7380143b7fcba60ad391d9cf8c55ea67.jpg')

print(im.shape)

# create a grayscale version
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(gray.shape)
