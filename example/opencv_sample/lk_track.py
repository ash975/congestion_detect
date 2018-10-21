#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
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

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.1,     # 要根据场景难度进行调整
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):
        segement_dynamic = cv2.imread('../../../assets/segements/segement_dynamic.png', 0)
        self.track_len = 5
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        self.mask = segement_dynamic  # cv2.threshold(segement_dynamic, 127, 255, cv2.THRESH_BINARY)

        self.n = 0
        self.m = 0
        self.sum_p = 0
        self.result_p = [0, 0, 0, 0]
        self.t_1 = [0]
        self.t_2 = [0]
        self.t_3 = [0]
        self.d_1 = [0]
        self.d_2 = [0]
        self.d_3 = [0]
        self.sum_t1 = 0
        self.sum_t2 = 0
        self.sum_t3 = 0
        self.result_save = {}
        self.result_temp = []
        self.is_empty = False
        self.anglized = 360 / (2 * np.pi)
        self.color = np.random.randint(0, 255, (100, 3))
        self.is_crowded = False
        self.detect = []

    def run(self):

        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=self.mask)
            # cv2.imshow('frame_gray', frame_gray)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)

                # 自己加的
                # 计算向量方向, 建立坐标系, 建立的坐标系是向左为x正(0度) 上y正
                V_p = (p0r - p1)  # .reshape(-1, 2)
                # x_base = np.array([[0], [1]])
                x_base = np.array([[1, 0]])
                # muti = V_p.dot(x_base).reshape(V_p.shape[0])
                muti = np.inner(V_p, x_base.reshape(-1,1,2)).reshape(V_p.shape[0])
                # print('muti', muti[0] , 'Vp', V_p[0], V_p[0].shape,  'x_base', x_base, x_base.shape)
                norm_vp = np.zeros(V_p.shape[0])
                for i, v in enumerate(V_p):
                    norm_vp[i] = np.linalg.norm(v)
                cos = muti / norm_vp
                angle = np.arccos(cos) * self.anglized

                # good = (d < 1) * ((angle < 20) + (angle > 160))  # 筛选出需要的点 : 东西车道
                # good = (d < 1) * ((angle > 20) * (angle < 160))  # 筛选出需要的点 : 南北车道
                good = (d < 1) * ((angle > 15) * (angle < 165))  # 筛选出需要的点 : 南北车道 ( 101 效果好)


                new_tracks = []
                self.t_1 = [0, ]
                self.d_1 = [0, ]
                self.t_2 = [0, ]
                self.d_2 = [0, ]
                self.t_3 = [0, ]
                self.d_3 = [0, ]
                for d_i, tr, (x, y), good_flag in zip(norm_vp, self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))

                    # 判断区域
                    if x > 509:
                        if y > 379:
                            self.t_1.append((x, y))
                            self.d_1.append(d_i)

                    if x > 322 and x < 1070:
                        if y > 203 and y < 337:
                            self.t_2.append((x, y))
                            self.d_2.append(d_i)

                    if x > 300 and x < 640:
                        if y > 138 and y < 208:
                            self.t_3.append((x, y))
                            self.d_3.append(d_i)
                    # 自己加的

                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                # cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, self.color[np.random.randint(100)].tolist()) # default (0, 255, 0) 绿色
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0)) # default (0, 255, 0) 绿色
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                draw_str(vis, (1020, 20), 'frame: %d' % self.frame_idx)

                # 计算车速
                if not self.is_empty:
                    if self.n <= 10:
                        self.sum_p += np.mean(norm_vp)
                        self.sum_t1 += np.mean(self.d_1)
                        self.sum_t2 += np.mean(self.d_2)
                        self.sum_t3 += np.mean(self.d_3)
                        self.n += 1
                    else:
                        self.result_p = [self.sum_p, self.sum_t1, self.sum_t2, self.sum_t3]
                        self.result_save.update({self.frame_idx: self.result_p})
                        self.n = 0
                        self.sum_p = 0
                        self.sum_t1 = 0
                        self.sum_t2 = 0
                        self.sum_t3 = 0

                # 判断拥堵
                if self.n == 0 and self.m < 1800:
                    self.result_temp += self.result_p
                    self.m += 1
                elif self.m == 1800:
                    self.detect = self.result_temp / 1800
                    if self.detect[0] <= 15:
                        self.is_crowded = True
                    else:
                        self.is_crowded = False


                # 打印到屏幕上
                draw_str(vis, (1020, 40), 't_f speed: %d' % int(self.result_p[0]))
                draw_str(vis, (1020, 60), 't_1 speed: %d' % int(self.result_p[1]))
                draw_str(vis, (1020, 80), 't_2 speed: %d' % int(self.result_p[2]))
                draw_str(vis, (1020, 100), 't_3 speed: %d' % int(self.result_p[3]))
                draw_str(vis, (1020, 120), 'is_empty: %s' % str(self.is_empty))
                draw_str(vis, (500, 300), 'Crowded: %s' % str(self.is_crowded))

            if self.frame_idx % self.detect_interval == 0:
                # mask = np.zeros_like(frame_gray)
                # mask[:] = 255
                mask = self.mask.copy()
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

                if len(self.tracks) < 50:
                    self.is_empty = True
                else:
                    self.is_empty = False


            # if self.frame_idx % 500 == 0:
            #     result_file = open('result_file_101_0828_2', 'w')
            #     result_file.write(json.dumps(self.result_save))
            #     result_file.close()


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            # cv2.imshow('mask', mask)

            ch = cv2.waitKey(1)
            if ch == 27:
                break


def main():
    # import sys
    # try:
    #     video_src = sys.argv[1]
    # except:
    #     video_src = 0

    video_src = '../../../assets/101_合并文件.avi'
    # segement_dynamic = cv2.imread('../../../assets/segements/segement_dynamic.png', 0)
    # _, segement_dynamic = cv2.threshold(segement_dynamic, 127, 255, cv2.THRESH_BINARY)

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
