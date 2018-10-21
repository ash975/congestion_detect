import numpy as np
import cv2


########

# def draw_circle(img, event,x,y,flags,param):
#   if event==cv2.EVENT_MOUSEMOVE:
#     cv2.circle(img,(x,y),100,(255,0,0),-1)

#######

# 保存地址
background_img = '../../assets/background/102/background'
f_vedio = '../../assets/102_绿灯.avi'

cap = cv2.VideoCapture(f_vedio)

# fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg1 = cv2.createBackgroundSubtractorMOG2()
# fgbg.setDetectShadows(False)
fgbg1.setDetectShadows(False)
# fgbg.setVarThreshold(60)
fgbg1.setVarThreshold(60)

# 对背景建模
ret, frame = cap.read()
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

road_shape = background.shape
road = np.zeros(road_shape,np.uint8)
n = 1
flag = True

while(1):
    ret, frame = cap.read()
    n += 1

    # 灰度, 有效减少噪声
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if flag:
        # fgmask = fgbg.apply(gray_frame, 1)
        # 用来更新背景
        fgmask1 = fgbg1.apply(gray_frame, 1)
        # 获取MOG2的背景 有点慢
        # background = fgbg1.getBackgroundImage()

    # masked = cv2.bitwise_and(frame, frame, mask=fgmask)

    # 图像显示
    # cv2.imshow('fgmask', fgmask)
    # cv2.imshow('masked', masked)
    # cv2.imshow('background', background)
    # cv2.imshow('frame', frame)

    if n%200 == 0:
        background = fgbg1.getBackgroundImage()
        cv2.imwrite(background_img + str(n) + '.png', background)

    # 终止
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif not ret:
        break
    elif k == ord('s'):       # press 's' save and exit
        cv2.imwrite(background_img+str(n)+'.png' , background)
        # cv2.imwrite('../../assets/frame' + str(n) + '.png', frame)
    elif k == ord('r'):
        flag = True
    elif k == ord('t'):
        flag ==  False

cap.release()
cv2.destroyAllWindows()