import numpy as np
import cv2

# 打开文件设置
f_video_101 = '../assets/101_合并文件.avi'
f_udp = 'udp://@192.168.1.1:1234'
f_video_test = '../assets/test_demo/20170811071506-3701033120-101.AVI'
f_video_18 = '../assets/101_18.avi'
f_video_8 = '../assets/101_8.avi'
f_video_102 = '../assets/102_合并文件.avi'
f_video_101_718 = '../assets/101_718.avi'
f_video = f_video_101_718


# 保存文件设置
# 路径 + 文件名 ，程序内自动编号 格式 png
f_background_101 = '../assets/background/101/101_f_bg'
f_background_102 = '../assets/background/102/102_f_bg'
f_background = f_background_101

# 初始化变量
n_frame = 0
n_mog2 = 0
n_signal = 0
signal = False
begin_detect = False
is_jam = False
level_jam = 0
n_jam = 0
a_jam = np.zeros((300, 1))
p_jam = 0
p_dynamic = 0
p_static = 0
n_high_jam = 0

# 载入视频
cap = cv2.VideoCapture(f_video)

# 载入 segement
segement_dynamic_101 = '../assets/segements/segement_dynamic.png'
segement_dynamic = cv2.imread('../assets/segements/segement_dynamic.png', 0)
segement_static = cv2.imread('../assets/segements/segement_static_3.png', 0)
segement_signal_green = cv2.imread('../assets/segements/segement_signal_green.png', 0)

segement_dynamic = cv2.imread('../assets/segements/segement_lk_101.png', 0)
segement_static = cv2.imread('../assets/segements/segement_lk_101.png', 0)

# mask 二值化
_, segement_dynamic = cv2.threshold(segement_dynamic, 127, 255, cv2.THRESH_BINARY)
_, segement_static = cv2.threshold(segement_static, 127, 255, cv2.THRESH_BINARY)
_, segement_signal_green = cv2.threshold(segement_signal_green, 127, 255, cv2.THRESH_BINARY)

# 初始化MOG 动态监测
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg1 = cv2.createBackgroundSubtractorMOG2()
fgbg.setDetectShadows(False)
fgbg.setVarThreshold(60)

# 载入预置背景
background = cv2.imread('../assets/background/101_bg1000.png', 0)
# 初始化mog背景
background_mog = background

# 载入视频图像大小
img_shape = background.shape

while (1):
    ## 读取视频
    ret, frame = cap.read()

    ## 终止条件
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif not ret:
        break

    ## 灰度化, 有效减少噪声
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 记录读入帧数
    n_frame += 1

    # 检测绿色信号灯
    frame_signal_green = cv2.bitwise_and(frame, frame, mask=segement_signal_green)
    _, signal_green = cv2.threshold(frame_signal_green, 240, 255, cv2.THRESH_BINARY)
    # cv2.imshow('signal_green', signal_green)
    # print(np.sum(signal_green))
    if np.sum(signal_green) >= 40000:
        signal = True
    else:
        signal = False

    # 绿灯 - 动态处理
    if signal:
        # MOG 识别动态物体
        fgmask = fgbg.apply(gray_frame, 1)
        # 记录背景处理数
        n_mog2 += 1
        # MOG替换预置背景
        if n_mog2 % 30 == 0:
            # 获取MOG2的背景
            background_mog = fgbg.getBackgroundImage()
            background = background_mog
        # 存背景图片
        if n_mog2 % 30 == 0:
            cv2.imwrite(f_background + str(n_mog2) + '.png', background)
        # cv2.imshow('background_mog', background_mog)
        # roi
        roi_dynamic = cv2.bitwise_and(fgmask, fgmask, mask=segement_dynamic)
        ## 对 roi_dynamic 进行处理填充
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        # roi_dynamic = cv2.morphologyEx(roi_dynamic, cv2.MORPH_CLOSE, kernel)
        # roi_dynamic = cv2.erode(roi_dynamic, None, iterations=2)
        # roi_dynamic = cv2.dilate(roi_dynamic, None, iterations=2)

        ## 计算占道比
        # c_segement_dynamic = np.sum(segement_dynamic) + 1
        # c_dynamic = np.sum(roi_dynamic)
        # p_dynamic = c_dynamic / c_segement_dynamic
        # cv2.imshow('roi_dynamic', roi_dynamic)

        ## 检测角
        # corners = cv2.goodFeaturesToTrack(roi_dynamic, 25, 0.01, 10)
        # corners = np.int0(corners)

        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(frame, (x, y), 3, 255, -1)

    ## 背景是否初始化完成
    if n_mog2 >= 30:
        ## 开始监测
        begin_detect = True

    ## 红灯 - 静态处理
    if begin_detect:
        ## roi
        roi_static = cv2.bitwise_and(frame, frame, mask=segement_static)
        roi_background = cv2.bitwise_and(background, background, mask=segement_static)
        ## 灰度, 有效减少噪声
        roi_static_gray = cv2.cvtColor(roi_static, cv2.COLOR_BGR2GRAY)
        ## 作差取得物体
        substact_static = roi_static_gray - roi_background
        ## 二值化 dst
        _, substact_static = cv2.threshold(substact_static, 20, 255, cv2.THRESH_BINARY)
        cv2.imshow('substact_static_t', substact_static)
        # cv2.imshow('substact_static', substact_static)
        ## 对二值化 dst 进行处理填充 去噪声
        substact_static = cv2.erode(substact_static, None, iterations=1)
        substact_static = cv2.dilate(substact_static, None, iterations=2)
        # 计算比率
        c_static = np.sum(segement_static) + 1
        c_dst = np.sum(substact_static)
        p_static = c_dst / c_static
        cv2.imshow('substact_static', substact_static)

        ## 判别拥堵
        if n_jam >=0 and n_jam <= 299:
            a_jam[n_jam] = p_static
            n_jam += 1
        elif n_jam == 300:
            p_jam = np.sum(a_jam) / 300
            n_jam = 0
            if p_jam >= 0.7:
                n_high_jam += 1
                if n_high_jam >=5:
                    is_jam = True
                    level_jam = 3
                    cv2.imwrite('../assets/jam/101_f_jam_fram'+ str(n_frame) + '.png', frame)
                    cv2.imwrite('../assets/jam/101_f_jam_bg' + str(n_frame) + '.png', background)
                    cv2.imwrite('../assets/jam/101_f_jam_substact_static' + str(n_frame) + '.png', substact_static)
            elif p_jam >= 0.3:
                level_jam = 2
                is_jam = False
                n_high_jam = 0
            else:
                level_jam = 1
                is_jam = False
                n_high_jam = 0

    ## 判定级别
    if level_jam == 0:
        level_jam_discrib = 'initial'#'初始化'
    elif level_jam == 1:
        level_jam_discrib = 'I' #'通畅'
    elif level_jam == 2:
        level_jam_discrib = 'II' # '缓行'
    elif level_jam == 3:
        level_jam_discrib = 'III' #'拥堵'


    # 前景
    # fg_mask = cv2.bitwise_and(frame, frame, mask=fgmask)

    # 记录数据
    # print('p_dynamic', p_dynamic)
    # print('p_static', p_static)

    # 打印数据文字到视频上
    x = int(img_shape[0] / 20 ) * 30
    y = int(img_shape[1] / 80)

    # cv2.putText(frame, 'p_dynamic: ' + str(p_dynamic), (x, y), 0, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, 'p_static: ' + str(p_static), (x, y + 20 * 1), 0, 0.5, (0, 255, 255), 0)
    cv2.putText(frame, 'signal: ' + str(signal), (x, y + 20 * 2), 0, 0.5, (0, 255, 255), 0)
    cv2.putText(frame, 'n_frame: ' + str(n_frame), (x, y + 20 * 3), 0, 0.5, (0, 255, 255), 0)
    cv2.putText(frame, 'n_mog2: ' + str(n_mog2), (x, y + 20 * 4), 0, 0.5, (0, 255, 255), 0)
    cv2.putText(frame, 'p_jam: ' + str(p_jam), (x, y + 20 * 5), 0, 0.5, (0, 255, 255), 0)
    cv2.putText(frame, 'is_jam: ' + str(is_jam), (x, y + 20 * 6), 0, 0.5, (0, 255, 255), 0)
    cv2.putText(frame, 'level: ' + level_jam_discrib, (x, y + 20 * 7), 0, 0.5, (0, 255, 255), 0)


    # 图像显示
    cv2.imshow('frame', frame)
    cv2.imshow('background', background)
    # cv2.imshow('background_mog', background_mog)
    # cv2.imshow('signal', signal_green)
    # cv2.imshow('roi_dynamic', roi_dynamic)

cap.release()
cv2.destroyAllWindows()
