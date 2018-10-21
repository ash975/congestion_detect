import numpy as np
import cv2
import matplotlib.pyplot as plt




########

# def draw_circle(img, event,x,y,flags,param):
#   if event==cv2.EVENT_MOUSEMOVE:
#     cv2.circle(img,(x,y),100,(255,0,0),-1)

#######

cap = cv2.VideoCapture('../../assets/test_demo/20170811071506-3701033120-101.AVI')
# cap = cv2.VideoCapture('../../assets/101_合并短.avi')

# mask
segements1 = cv2.imread('../../assets/segements.png', 0)
segement_signal_green = cv2.imread('../../assets/segements/segement_signal_green.png', 0)
# mask 二值化
ret1, segements = cv2.threshold(segements1, 127, 255, cv2.THRESH_BINARY)
ret, segement_signal = cv2.threshold(segement_signal_green, 127, 255, cv2.THRESH_BINARY)

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg1 = cv2.createBackgroundSubtractorMOG2()
fgbg.setDetectShadows(False)
fgbg.setVarThreshold(60)

# 对背景建模
ret, frame = cap.read()
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

road_shape = background.shape
road = np.zeros(road_shape,np.uint8)
signal = False

while(1):
    ret, frame = cap.read()

    # ROI
    # ret1, segements = cv2.threshold(segements1, 127, 255, cv2.THRESH_BINARY)
    roi = cv2.bitwise_and(frame, frame, mask=segements)

    # 灰度, 有效减少噪声
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 检测绿色信号灯
    frame_signal_green = cv2.bitwise_and(frame, frame, mask=segement_signal)
    cv2.imshow('signal', frame_signal_green)
    _, signal_green = cv2.threshold(frame_signal_green, 240, 255, cv2.THRESH_BINARY)
    cv2.imshow('signal_green', signal_green)
    print(np.sum(signal_green))
    if np.sum(signal_green) >= 70000:
        signal = True
    else:
        signal = False

    # MOG 识别动态物体
    fgmask = fgbg.apply(gray_frame, 1)

    # 用来更新背景
    # fgmask1 = fgbg1.apply(gray_frame, 1)
    # 获取MOG2的背景 有点慢
    # background = fgbg1.getBackgroundImage()

    # 对 mask 进行处理填充
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)


    # 前景
    # masked = cv2.bitwise_and(frame, frame, mask=fgmask)
    # masked = cv2.bitwise_and(frame, frame, mask=closed)

    # 绘制移动轨迹
    # gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # road = cv2.bitwise_or(gray_masked, road)


    # 检测圆形
    # circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=50, minRadius=55, maxRadius=80)
    # rows = gray_frame.shape[1]
    # circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 2, rows, 200, 100)
    # center = (int(circles.tolist()[0][0][0]),int(circles.tolist()[0][0][1]))
    # radius = int(circles.tolist()[0][0][2])
    # cv2.circle(frame, center, radius, (0,0,255), -1)

    # 打印一些文字到视频上
    # x = int(masked.shape[0] / 2)
    # y = int(masked.shape[1] / 2)
    # backgroundratio = fgbg.getBackgroundRatio()
    #
    # cv2.putText(masked, 'backgroundratio: ' + str(backgroundratio), (x, y), 0, 0.5, (0, 0, 255), 2)


    # 图像显示
    # cv2.imshow('fgmask', fgmask)
    # cv2.imshow('masked', masked)
    # cv2.imshow('background', background)
    # cv2.imshow('frame', frame)
    # cv2.imshow('road', road)
    # cv2.imshow('roi', roi)
    # cv2.imshow('closed', closed)

    # 终止
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif not ret:
        break

cap.release()
cv2.destroyAllWindows()