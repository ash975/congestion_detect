
# 读写图片
if False:
    import cv2

    # img = cv2.imread('./resources/res1.jpg', 0)     # 0 - gray; 1 - RGB
    img = cv2.imread('./resources/res1.jpg')        # default RGB
    img_gray = cv2.imread('./resources/res1.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_bgr = cv2.imread('./resources/res1.jpg', cv2.CV_LOAD_IMAGE_COLOR)
    img_unchanged = cv2.imread('./resources/res1.jpg', cv2.CV_LOAD_IMAGE_UNCHANGED)

    cv2.imshow('image', img)
    key = cv2.waitKey(0)
    if key == 27:       # press 'ESC' to exit
        cv2.destroyAllWindows()
    elif key == ord('s'):       # press 's' save and exit
        cv2.imwrite('./resources/res1gray.jpg', img)
        cv2.destroyAllWindows()

# 获取图片属性
if False:
    import cv2

    img = cv2.imread('./resources/res1.jpg')
    # 很多错误会出在 dtype 上
    print('shape:', img.shape, '\nsize:', img.size, '\ndtype:', img.dtype)

# 输出文本
if False:
    import cv2

    img = cv2.imread('./resources/res1.jpg')

    x = int(img.shape[0] / 2)
    y = int(img.shape[1] / 2)

    cv2.putText(img, 'Hello', (x, y), 0, 0.5,  (0,0,255), 2)
    cv2.imshow('image', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

# 缩放图片
if False:
    import cv2
    import numpy as np

    img = cv2.imread('./resources/res1.jpg')
    resized1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image1', resized1)

    # method 2
    height, width = img.shape[:2]
    resized2 = cv2.resize(img, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image2', resized2)
    key = cv2.waitKey(0)
    if key == 27:  # press 'ESC' to exit
        cv2.destroyAllWindows()

# 图像平移
if False:
    import cv2
    import numpy as np

    img = cv2.imread('./resources/res1.jpg')
    rows, cols = img.shape[0:2]

    # dst(x,y)=src(M11x+M12y+M13,M21x+M22y+M23)
    M = np.float32([[1,0,100], [0,1,50]])
    dst = cv2.warpAffine(img, M, (cols,rows))
    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图像旋转
if False:
    import cv2

    img = cv2.imread('./resources/res1.jpg')
    rows, cols = img.shape[0:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    print(M)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 仿射变换
if False:
    import cv2
    import numpy as np

    img = cv2.imread('./resources/res1.jpg')
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图像颜色变换
# 所有的变化条件: flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
if False:
    import cv2

    img = cv2.imread('./resources/res1.jpg')
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 识别摄像头视频中的蓝色部分
if False:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)

    while (1):

        # 读取视频的每一帧
        _, frame = cap.read()

        # 将图片从 BGR 空间转换到 HSV 空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义在HSV空间中蓝色的范围
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 根据以上定义的蓝色的阈值得到蓝色的部分
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

# 通道的拆分/合并处理
if False:
    import cv2

    img = cv2.imread('./resources/res1.jpg')
    b, g, r = cv2.split(img)

    cv2.imshow('b', r)

    img = cv2.merge((b, g, r))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图片添加边距( padding )
if False:
    import cv2

    BLUE = [255, 0, 0]

    img1 = cv2.imread('./resources/res1.jpg')

    replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)

    cv2.imshow('img', constant)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 合并两个图片 ( 相加 )
# error: (-209) The operation is neither 'array op array'
# (where arrays have the same size and the same number of channels),
# nor 'array op scalar', nor 'scalar op array' in function cv::arithm_op
if False:
    import cv2

    img2 = cv2.imread('./resources/res1.jpg')
    img1 = cv2.imread('./resources/res2.png')

    dst = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 位操作合并两个图片, mask去背景 再加到第二个图上

if True:
    # Load two images
    import cv2

    img2 = cv2.imread('./resources/opencv-logo-white.png')
    img1 = cv2.imread('./resources/res3.jpg')

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)
    cv2.imshow('img1_bg', img1_bg)
    cv2.imshow('img2_fg', img2_fg)
    cv2.imshow('mask', mask)
    cv2.imshow('mask_inv', mask_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

