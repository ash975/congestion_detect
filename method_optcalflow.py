#!/usr/bin/env python

'''
城市交叉路拥堵检测
===============

源自 OpenCV 例程: Lucas-Kanade tracker,
使用 goodFeatureToTrack 对追踪线初始化,
back-tracking 进行校准.

Dependence
----------


Usage
-----

Keys
-----
ESC - exit

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock
import json

# lk 光流法参数设置
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 本应用整体设置
feature_params = dict(maxCorners=500,
                      qualityLevel=0.1,
                      minDistance=15,
                      blockSize=15,
                      useHarrisDetector=False)

# 弧度制角度化制
ANGLIZED = 360 / (2 * np.pi)


class App:
    def __init__(self, video_src, segement_dynamic):
        self.track_len = 5  # 跟踪轨迹的长度
        self.detect_interval = 5  # 检测区间
        self.tracks = []  # 跟踪轨迹数组
        self.frame_idx = 0  # 视频帧数
        self.color = np.random.randint(0, 255, (100, 3))  # 追踪点颜色

        self.mask = segement_dynamic  # 蒙版 获得 ROI
        self.cam = video.create_capture(video_src)  # 视频源
        self.prev_gray = None

        self.n = 0  # 循环控制数 n   10 帧 1 个小循环
        self.m = 0  # 循环控制数 m   1800 帧 1 个大循环
        self.sum_p = 0  # 总体距离值
        self.result_p = [0, 0, 0, 0]  # 距离值数组
        self.d_p = [0, ]  # 全体 单位距离数组 (小循环)
        self.d_1 = [0, ]  # 区域1 单位距离值数组 (小循环)
        self.d_2 = [0, ]  # 区域2 单位距离值数组 (小循环)
        self.d_3 = [0, ]  # 区域3 单位距离值数组 (小循环)
        self.sum_t1 = 0  # 区域1 单位时间距离值总和
        self.sum_t2 = 0  # 区域2 单位时间距离值总和
        self.sum_t3 = 0  # 区域3 单位时间距离值总和

        self.sums = [self.sum_p, self.sum_t1, self.sum_t2, self.sum_t3]
        self.ds = [self.d_p, self.d_1, self.d_2, self.d_3]

        self.result_save = {}  # 储存结果, 做数据分析
        self.result_temp = [0, 0, 0, 0]  # 临时结果, 用来判断拥堵

        self.is_empty = False  # 道路是否为空
        self.detect = []  # 检测车辆
        self.is_crowded = False  # 拥堵标志量

        self.fps = 0


    def run(self):

        while True:
            # 读视频帧, ret 表示有无
            ret, frame = self.cam.read()
            # 灰度化
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 二值化等处理
            frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=self.mask)
            # 显示一下
            # cv2.imshow('frame_gray', frame_gray)
            # 拷贝原帧, 方便操作
            vis = frame.copy()

            # 打好追踪点, 进行追踪
            if len(self.tracks) > 0 and not self.is_empty:
                # 取前一帧 和 当前帧
                img0, img1 = self.prev_gray, frame_gray
                # 获取最新的追踪点
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # 使用光流法计算追踪点的当前点
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # back-tracking 矫正
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                # 保留误差值小于标准误差的点
                deviation = abs(p0 - p0r).reshape(-1, 2).max(-1)

                '''
                计算向量方向, 建立坐标系, 左上角为(0, 0), 向右x正(0度) 下y正
                
                '''
                # 计算追踪点移动的距离
                vector_ab = (p1 - p0)
                # 求L2范数 |A|
                norm_vp = np.zeros(vector_ab.shape[0])
                for i, v in enumerate(vector_ab):
                    norm_vp[i] = np.linalg.norm(v)
                x_base = np.array([[1, 0]])  # x轴单位向量
                inner_muti = np.inner(vector_ab, x_base.reshape(-1, 1, 2)).reshape(vector_ab.shape[0])

                # cos = inner_muti / norm_vp # 这样做会有除0问题，下面的是解决方法：使用where条件筛掉等于0的
                cos = np.true_divide(inner_muti, norm_vp, out=np.zeros_like(inner_muti), where=norm_vp != 0)
                angle = np.arccos(cos) * ANGLIZED

                # filter_array 为筛选矩阵
                # filter_array = (deviation < 1) * ((angle < 20) + (angle > 160))  # 筛选出需要的点 : 东西车道
                # filter_array = (deviation < 1) * ((angle < 20) + (angle > 130))  # 筛选出需要的点 : 东西车道
                # filter_array = (deviation < 1) * ((angle > 20) * (angle < 160))  # 筛选出需要的点 : 南北车道
                filter_array = (deviation < 0.1) * ((angle > 25) * (angle < 155))  # 筛选出需要的点 : 南北车道

                # 初始化变量
                for i in range(0, len(self.ds)):
                    self.ds[i] = [0, ]
                new_tracks = []

                for distance, track, (x, y), good_flag in zip(norm_vp, self.tracks, p1.reshape(-1, 2), filter_array):
                    if not good_flag:
                        continue
                    track.append((x, y))

                    # 全局的点
                    self.ds[0].append(distance)

                    # 判断区域
                    if x > 509 and y > 379:
                        self.ds[1].append(distance)
                    elif 322 < x < 1070 and 203 < y < 337:
                        self.ds[2].append(distance)
                    elif 300 < x < 640 and 138 < y < 208:
                        self.ds[3].append(distance)

                    if len(track) > self.track_len:
                        del track[0]

                    new_tracks.append(track)

                    # 标出追踪点 ( 更明显 )
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                self.tracks = new_tracks

                # 炫光霓虹追踪线
                # cv2.polylines(vis, [np.int32(track) for track in self.tracks],
                #               False, self.color[np.random.randint(100)].tolist())

                # 原谅绿追踪线
                # cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))  # default (0, 255, 0) 绿色

                # 计算车速
                if self.n <= 12:
                    for i in range(len(self.sums)):
                        self.sums[i] += np.mean(self.ds[i])
                    self.n += 1
                else:
                    self.result_p = self.sums.copy()
                    self.result_save.update({self.frame_idx: self.result_p[0]})
                    for i in range(len(self.sums)):
                        # self.result_p[i] = self.sums[i]
                        self.sums[i] = 0
                    self.n = 0

                # 判断拥堵
                # if self.n == 0 and self.m < 40:
                if self.n == 0 and self.m < 10:
                    for i in range(len(self.result_p)):
                        self.result_temp[i] += self.result_p[i]
                    self.m += 1
                # elif self.m == 40:
                elif self.m == 10:
                    self.m = 0
                    # self.detect = self.result_temp[0] / 40
                    self.detect = self.result_temp[0] / 10
                    self.result_temp = [0, 0, 0, 0]

                    if 30 < self.detect <= 60:
                        cv2.imwrite('./out/result/low_than_60_vis_' + str(self.frame_idx) + '.jpg', vis)
                        #cv2.imwrite('./out/result/low_than_60_mask_' + str(self.frame_idx) + '.jpg', mask)

                    if self.detect <= 30:
                        cv2.imwrite('./out/result/low_than_30_vis_' + str(self.frame_idx) + '.jpg', vis)
                        #cv2.imwrite('./out/result/low_than_30_mask_' + str(self.frame_idx) + '.jpg', mask)
                        self.is_crowded = True
                    else:
                        self.is_crowded = False

            # 间歇寻找追踪点, 使用 goodFeaturesToTrack, frame_idx (视频当前帧数) % detect_interval (检测间隔帧数)
            if self.frame_idx % self.detect_interval == 0:

                # 尝试 全白 和 全黑 蒙版
                # mask = np.zeros_like(frame_gray)
                # mask[:] = 255
                # 使用初始化的蒙版
                mask = self.mask.copy()

                # 在蒙版上画上当前追踪轨迹的点
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)

                # 寻找新的特征点并加入追踪点
                point = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                # point = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
                if point is not None:
                    for x, y in np.float32(point).reshape(-1, 2):
                        self.tracks.append([(x, y)])

                # 如果追踪点小于 100 个, 则判断无车
                if len(self.tracks) < 100:
                    self.is_empty = True
                else:
                    self.is_empty = False

            # 显示到屏幕上

            if True:
                img_print = vis
                # draw_str(img_print, (1020, 40), 't_f speed: %d' % self.result_p[0])
                # draw_str(img_print, (1020, 60), 't_1 speed: %d' % self.result_p[1])
                # draw_str(img_print, (1020, 80), 't_2 speed: %d' % self.result_p[2])
                # draw_str(img_print, (1020, 100), 't_3 speed: %d' % self.result_p[3])
                # draw_str(img_print, (1020, 120), 'is_empty: %s' % self.is_empty)
                # draw_str(img_print, (500, 300), 'Crowded: %s' % self.is_crowded)
                # draw_str(img_print, (20, 20), 'track count: %d' % len(self.tracks))
                # draw_str(img_print, (1020, 20), 'frame: %d' % self.frame_idx)

                # 在蒙版上画上当前追踪轨迹的点
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                cv2.polylines(img_print, [np.int32(track) for track in self.tracks], False,
                              (0, 255, 0))  # default (0, 255, 0) 绿色

            # 间歇保存数据, 用作分析
            # if self.frame_idx % 120 == 0:
            #     self.result_file = open('result_file_101_0601', 'w')
            #     # result_file = open('result_file_101_0601', 'w')
            #     self.result_file.write(json.dumps(self.result_save))
            #     # self.result_save = {}
            #     self.result_file.close()

            # 记录帧数
            self.frame_idx += 1
            # 保存前一帧
            self.prev_gray = frame_gray
            # 显示追踪
            cv2.imshow('lk_track', vis)
            # 显示蒙版
            # cv2.imshow('mask', mask)

            # 等待 ESC 退出
            ch = cv2.waitKey(1)
            if ch == 27:
                # self.videoWriter.release()
                break
            if ch == 112:
                cv2.imwrite('./out/screenshot_origin_' + str(self.frame_idx) + '.jpg', vis)
                #cv2.imwrite('./out/screenshot_mask_' + str(self.frame_idx) + '.jpg', mask)


def main():
    """
    作为 app 运行, 读取系统参数作为视频源
    :return:
    """
    # import sys
    # try:
    #     video_src = sys.argv[1]
    # except:
    #     video_src = 0

    '''
    调试使用, 从文件选择视频源
    '''
    # 从文件选择视频源
    video_src = '../assets/101_clip.avi'
    # 选择ROI
    # segement_dynamic = cv2.imread('../assets/segements/401/segement_401_1.png', 0)
    segement_dynamic = cv2.imread('../assets/segements/segement_lk_101.png', 0)
    # _, segement_dynamic = cv2.threshold(segement_dynamic, 127, 255, cv2.THRESH_BINARY)    # 对蒙版的处理

    # print(__doc__)
    import time
    t1 = time.time()
    try:
        App(video_src, segement_dynamic).run()
        cv2.destroyAllWindows()
    except Exception as e:
        pass
    t2 = time.time()

    print(t2-t1)


if __name__ == '__main__':
    main()
