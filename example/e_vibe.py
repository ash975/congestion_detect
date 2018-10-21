import random
import numpy as np
import cv2


class ViBe:
    '''''
    classdocs
    '''
    __defaultNbSamples = 20  # 每个像素点的样本个数
    __defaultReqMatches = 2  # min指数
    __defaultRadius = 20  # Sqthere半径
    __defaultSubsamplingFactor = 16  # 子采样概率
    __BG = 0  # 背景像素
    __FG = 255  # 前景像素
    __c_xoff = [-1, 0, 1, -1, 1, -1, 0, 1, 0]  # x的邻居点 len=9
    __c_yoff = [-1, 0, 1, -1, 1, -1, 0, 1, 0]  # y的邻居点 len=9

    __samples = []  # 保存每个像素点的样本值,len defaultNbSamples+1
    __Height = 0
    __Width = 0

    def __init__(self, grayFrame):
        '''''
        Constructor
        '''
        self.__Height = grayFrame.shape[0]
        self.__Width = grayFrame.shape[1]

        for i in range(self.__defaultNbSamples + 1):
            self.__samples.insert(i, np.zeros((grayFrame.shape[0], grayFrame.shape[1]), dtype=grayFrame.dtype));

        self.__init_params(grayFrame)

    def __init_params(self, grayFrame):
        # 记录随机生成的 行(r) 和 列(c)
        rand = 0
        r = 0
        c = 0

        # 对每个像素样本进行初始化
        for y in range(self.__Height):
            for x in range(self.__Width):
                for k in range(self.__defaultNbSamples):
                    # 随机获取像素样本值
                    rand = random.randint(0, 8)
                    r = y + self.__c_yoff[rand]
                    if r < 0:
                        r = 0
                    if r >= self.__Height:
                        r = self.__Height - 1  # 行
                    c = x + self.__c_xoff[rand]
                    if c < 0:
                        c = 0
                    if c >= self.__Width:
                        c = self.__Width - 1  # 列
                    # 存储像素样本值
                    self.__samples[k][y, x] = grayFrame[r, c]
            self.__samples[self.__defaultNbSamples][y, x] = 0

    def update(self, grayFrame, frameNo):
        foreground = np.zeros((self.__Height, self.__Width), dtype=np.uint8)
        for y in range(self.__Height):  # Height
            for x in range(self.__Width):  # Width
                # 用于判断一个点是否是背景点,index记录已比较的样本个数，count表示匹配的样本个数
                count = 0
                index = 0
                dist = 0.0
                while (count < self.__defaultReqMatches) and (index < self.__defaultNbSamples):
                    dist = float(grayFrame[y, x]) - float(self.__samples[index][y, x])
                    if dist < 0: dist = -dist
                    if dist < self.__defaultRadius: count = count + 1
                    index = index + 1

                if count >= self.__defaultReqMatches:
                    # 判断为背景像素,只有背景点才能被用来传播和更新存储样本值
                    self.__samples[self.__defaultNbSamples][y, x] = 0

                    foreground[y, x] = self.__BG

                    rand = random.randint(0, self.__defaultSubsamplingFactor)
                    if rand == 0:
                        rand = random.randint(0, self.__defaultNbSamples)
                        self.__samples[rand][y, x] = grayFrame[y, x]
                    rand = random.randint(0, self.__defaultSubsamplingFactor)
                    if rand == 0:
                        rand = random.randint(0, 8)
                        yN = y + self.__c_yoff[rand]
                        if yN < 0: yN = 0
                        if yN >= self.__Height: yN = self.__Height - 1
                        rand = random.randint(0, 8)
                        xN = x + self.__c_xoff[rand]
                        if xN < 0: xN = 0
                        if xN >= self.__Width: xN = self.__Width - 1
                        rand = random.randint(0, self.__defaultNbSamples)
                        self.__samples[rand][yN, xN] = grayFrame[y, x]
                else:
                    # 判断为前景像素
                    foreground[y, x] = self.__FG
                    self.__samples[self.__defaultNbSamples][y, x] += 1
                    if self.__samples[self.__defaultNbSamples][y, x] > 50:
                        rand = random.randint(0, self.__defaultNbSamples)
                        if rand == 0:
                            rand = random.randint(0, self.__defaultNbSamples)
                            self.__samples[rand][y, x] = grayFrame[y, x]
        return foreground



cap = cv2.VideoCapture('../../assets/101_合并短.avi')

ret, frame = cap.read()
dst_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
fgbg = ViBe(dst_1)

while(1):
    ret, frame = cap.read()
    dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    foreground = fgbg.update(dst, ret)

    cv2.imshow('foreground',foreground)
    cv2.imshow('frame', frame)
    cv2.imshow('dst', dst)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()