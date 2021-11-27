import numpy as np
import pandas as pd
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt


def CalSystemMatrix(theta, pictureSize, projectionNum, delta):
    squareChannels = np.power(pictureSize, 2)
    totalPorjectionNum = len(theta) * projectionNum
    gridNum = np.zeros((totalPorjectionNum, 2 * pictureSize))
    gridLen = np.zeros((totalPorjectionNum, 2 * pictureSize))
    t = np.arange(-(pictureSize - 1) / 2, (pictureSize - 1) / 2+1)
    for loop1 in range(len(theta)):
        for loop2 in range(pictureSize):
            u = np.zeros((2 * pictureSize))
            v = np.zeros((2 * pictureSize))
            th = theta[loop1]
            if th == 90:
            #如果计算的结果超出了网格的范围，则立刻开始计算下一个射束
                if ((t[loop2] >= pictureSize / 2 * delta) or (t[loop2] <= - pictureSize / 2 * delta)):
                    continue
                kout = pictureSize * np.ceil(pictureSize/2 - t[loop2]/delta)
                kk = np.arange(kout - (pictureSize -1 ), kout+1)
                u[0:pictureSize] = kk
                v[0:pictureSize] = np.ones(pictureSize) * delta
                
            elif th==0:
                if (t[loop2] >= pictureSize / 2 * delta) or (t[loop2] <= -pictureSize / 2 * delta):
                    continue
                kin = np.ceil(pictureSize/2 + t[loop2] / delta)
                kk = np.arange(kin, (kin + pictureSize * pictureSize), step=pictureSize)
                u[0:pictureSize] = kk
                v[0:pictureSize] = np.ones(pictureSize) * delta

            else:
                if th>90:
                    th_temp = th - 90
                elif th<90:
                    th_temp = 90 - th
                th_temp = th_temp * np.pi / 180
                #计算束线的斜率和截距
                b = t / np.cos(th_temp)
                m = np.tan(th_temp)
                y1d = -(pictureSize / 2) * delta * m + b[loop2]
                y2d = (pictureSize / 2) * delta * m + b[loop2]
                #if (y1d < -pictureSize / 2 * delta and y2d < -pictureSize/2 * delta) or (y1d > pictureSize / 2 * delta and y2d > -pictureSize / 2 * delta):
                if (y1d < -pictureSize / 2 * delta and y2d < -pictureSize/2 * delta) or (y1d > pictureSize / 2 * delta and y2d > pictureSize / 2 * delta):
                    continue
                if (y1d <= pictureSize / 2 * delta and  y1d >= -pictureSize/2 * delta and y2d > pictureSize / 2 * delta):
                    yin = y1d
                    d1 = yin - np.floor(yin / delta) * delta
                    kin = pictureSize * np.floor(pictureSize / 2 - yin / delta) + 1
                    yout = pictureSize / 2 * delta
                    xout = (yout - b[loop2]) / m
                    kout = np.ceil(xout / delta) + pictureSize / 2

                elif (y1d <= pictureSize/2 * delta and y1d >= -pictureSize/2 * delta and y2d >= -pictureSize/2 * delta and y2d < pictureSize/2 * delta):
                    yin = y1d
                    d1 = yin - np.floor(yin/delta) * delta
                    kin = pictureSize * np.floor(pictureSize / 2 - yin / delta) + 1
                    yout = y2d
                    #1:
                    #2:xout = (yout - b[loop2]) / m
                    kout = pictureSize * np.floor(pictureSize/2 - yout/delta) + pictureSize

                elif (y1d < - pictureSize / 2 * delta and y2d > pictureSize / 2 * delta):
                    yin = - pictureSize / 2  * delta
                    xin = (yin - b[loop2]) / m
                    d1 = pictureSize / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]
                    kin = pictureSize * (pictureSize - 1) + pictureSize / 2 + np.ceil(xin / delta)
                    yout = pictureSize / 2 * delta
                    #error: xout = (yout / b[loop2])/m
                    xout = (yout - b[loop2]) / m
                    kout = np.ceil(xout / delta) + pictureSize / 2

                elif (y1d < - pictureSize / 2 * delta and y2d >= -pictureSize / 2 * delta and y2d < pictureSize / 2 * delta):
                    yin = -pictureSize / 2 * delta 
                    xin = (yin - b[loop2]) / m
                    d1 = pictureSize / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]
                    kin = pictureSize * (pictureSize - 1) + pictureSize / 2 + np.ceil(xin / delta)
                    yout = y2d
                    kout = pictureSize * np.floor(pictureSize / 2 - yout / delta) + pictureSize
                else:
                    continue
                #计算第i条射束穿过的像素的编号和长度
                k = kin
                c = 0
                d2 = d1 + m * delta
                while k >= 1 and k <= squareChannels:
                    if d1 >= 0 and d2 > delta:
                        u[c] = k
                        v[c] = (delta - d1) * np.sqrt(np.power(m, 2) + 1) / m
                        if k > pictureSize and k != kout:
                            k = k - pictureSize
                            d1 = d1 - delta
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 >= 0 and d2 == delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k>pictureSize and k != kout:
                            k = k - pictureSize + 1
                            d1 = 0
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 >= 0 and d2 < delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k!=kout:
                            k = k + 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 <= 0 and d2 >= 0 and d2 <= delta:
                        u[c] = k
                        v[c] = d2 * np.sqrt(np.power(m, 2) + 1) / m
                        if k != kout:
                            k = k + 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 <= 0 and d2 > delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1) / m
                        if k > pictureSize and k != kout:
                            k = k - pictureSize
                            #k = k + 1
                            d1 = -delta + d1
                            d2 = d1 + m * delta
                        else:
                            break
                    else:
                        print(d1, d2, "error!!!")
                    c = c + 1
                #如果投影角度小于90度，应该利用投影射线关于y轴的对称性计算出权重因子向量。
                if th < 90:
                    u_temp = np.zeros(2 * pictureSize)
                    if u.any() == 0:
                        continue
                    indexULTZero = np.where(u>0)
                    for innerloop in range(len(u[indexULTZero])):
                        r = np.mod(u[innerloop], pictureSize)
                        if r == 0:
                            u_temp[innerloop] = u[innerloop] - pictureSize
                        else:
                            u_temp[innerloop] = u[innerloop] - 2 * r + pictureSize
                    u = u_temp
            gridNum[loop1 * projectionNum + loop2, :] = u
            gridLen[loop1 * projectionNum + loop2, :] = v
    return gridNum, gridLen

def DiscreteRadonTransform(image, theta):
    projectionNum = len(image[0])
    thetaLen = len(theta)
    radontansformRes = np.zeros((projectionNum, thetaLen), dtype='float64')
    for s in range(len(theta)):
        rotation = ndimage.rotate(image, -theta[s], reshape=False).astype('float64')
        radontansformRes[:, s] = sum(rotation)
    return radontansformRes
#适用于灰度图
image = cv2.imread("C:\\Users\\48853\\Desktop\\pic.png", cv2.IMREAD_GRAYSCALE)
#如果你的图片的色彩模式是RGB或者RGBA请使用以下语句
#如果使用上述imageio.imread，会出现数组维度不匹配的错误。
#image = cv2.imread("shepplogan.jpg", cv2.IMREAD_GRAYSCALE)

theta = np.linspace(0, 180, 60, dtype=np.float64)
#使用离散Radon变换获取投影值
projectionValue = DiscreteRadonTransform(image, theta)

#定义用到一些参数：旋转角度的矩阵，探测器的道数，图片尺寸，平移步长，最大迭代次数，驰豫因子
projectionNum = np.int64(256)
pictureSize = np.int64(256)
pictureSizeSquare = pictureSize * pictureSize
delta = np.int64(1)
irt_Num = np.int64(20)
lam = np.float64(0.25)

#计算投影矩阵
gridNum, gridLen = CalSystemMatrix(theta, pictureSize, projectionNum, delta)

dfgridNum = pd.DataFrame(gridNum)
dfgridLen = pd.DataFrame(gridLen)
#可以将系统矩阵存储到文件中,以后直接使用。
dfgridNum.to_csv("gridNum.csv", header=False, index=False)
dfgridLen.to_csv("gridLen.csv", header=False, index=False)

#gridNum = np.array(pd.read_csv("gridNum1.csv"), header=None)
#gridLen = np.array(pd.read_csv("gridLen1.csv"), header=None)
#存储重建获得的图像的矩阵
F = np.zeros((pictureSize*pictureSize, ))

irt_Num = 10
lam = 0.25
c = 0

#开始迭代过程
while(c < irt_Num):
    for loop1 in range(len(theta)):
        for loop2 in range(pictureSize):
            u = gridNum[loop1 * pictureSize + loop2, :]
            v = gridLen[loop1 * pictureSize + loop2, :]
            if u.any() == 0:
                continue
            w = np.zeros(pictureSizeSquare, dtype=np.float64)
            uLargerThanZero = np.where(u > 0)
            w[u[uLargerThanZero].astype(np.int64)-1] = v[uLargerThanZero]
            PP = w.dot(F)
            C = (projectionValue[loop2, loop1] - PP) / sum(np.power(w, 2)) * w.conj()
            F = F + lam * C
    F[np.where(F < 0)] = 0
    c = c + 1

F = F.reshape(pictureSize, pictureSize).conj()

#绘制Sinogram图和重建的结果
plt.subplot(1, 2, 1)
plt.imshow(projectionValue, cmap="gray")

plt.subplot(1, 2, 2)
plt.imshow(F, cmap="gray")
plt.savefig("modify90Sample.png", cmap="gray")
plt.show()