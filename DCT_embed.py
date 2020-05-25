from cv2 import cv2 as cv 
import numpy as np 
import matplotlib
import random,json

# 每个block的嵌入区，选最中间的中频8个位置
embedZone = [(0,7),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1),(7,0)]

def resize2octa(imgPath): 
    '''
    为了之后能够整划分8x8矩阵，此函数将图像risze成8x8的倍数
    '''
    imgRaw = cv.imread(imgPath,cv.IMREAD_UNCHANGED)
    rowSize = imgRaw.shape[0]
    columnSize = imgRaw.shape[1]
    # 要变成一个方阵
    adjust = rowSize if rowSize>columnSize else columnSize
    if adjust%8: # 等价于 adjust%8 != 0
        adjust = (1+adjust//8)*8
    size = (adjust,adjust)
    imgResized = cv.resize(imgRaw,size)
    return imgResized

def divideBlock(img):
    '''
    将图像分割成8x8的块，返回一个list存储了所有的块信息
    '''
    scale = img.shape[0] # resize过 rowSize 和 columnSize 一样，这里用scale表示
    columnSliceList = [ [column[i*8:(i+1)*8] for i in range(scale//8)] for column in img]
    blockList = []
    for i in range(scale//8):
        rowBlockList = columnSliceList[8*i:8*(i+1)]
        for x in range(scale//8):
            blockList.append([ rowBlockList[y][x] for y in range(8)])
    return blockList


def dctBlock(blockList):
    '''
    对每一个块进行dct变换
    '''
    # dct计算的矩阵是一个二维矩阵，所以直接把三维图像塞进来的话容易报错如上，
    # 如果要处理三维图像（三通道）可以把rgb三个维度一个一个丢进来dct，最后合成一个三维矩阵保存成彩色图像。
    # 本实验不用考虑以上这点，灰度图本来就是二维的
    # 同时，需要注意的是，要用np.float32把矩阵转换成32位浮点精度，这才是dct能处理的精度。所以必不可少。
    return [cv.dct(np.float32(block)) for block in blockList ]

def idctBlock(dctBlockList):
    '''
    对每一个块进行逆dct变换
    '''
    return [cv.idct(block) for block in dctBlockList]

def mergeBlock(blockList): # 注意此函数传入的必须是list不能是np的array，要先把array转成list
    scale = int(pow(len(blockList),0.5))*8
    imgMatrix = []
    for i in range(scale//8):
        for y in range(8):
            column = []
            for x in range(scale//8):
                column += blockList[(scale//8)*i+x][y]
            imgMatrix.append(column)
    return imgMatrix

# https://blog.csdn.net/missingu1314/article/details/8677703
def getEmbedPixel(imgEmbed,factor=0.05): # 缩放因子默认为0.05
    '''
    此函数用于将嵌入的水印图片的像素值，使用加法准则公式调节成嵌入的值，
    默认的缩放因子factor=0.05，关于此部分的介绍，详见实验报告
    '''
    rowScale = imgEmbed.shape[0]
    columnScale = imgEmbed.shape[1]
    pixelStream = []
    for i in range(rowScale):
        for j in range(columnScale): # 加法准则
            pixelStream.append(factor*imgEmbed.item(i,j))
    if len(pixelStream)%8 != 0:
        zeroAmount = 8 - len(pixelStream)%8
        for i in range(zeroAmount):
            pixelStream.append(0)
    return pixelStream

def getRandEmbedBlock(blockList,pixelStream):
    '''从blockList选几个block用来嵌入，函数返回一个存有藏了信息的block序号的list'''
    blockOrder = [i for i in range(len(blockList))]
    return random.sample(blockOrder,len(pixelStream)//8) # 完全随机
    # return blockOrder[:len(pixelStream)//8] # 从头一次取

def dctEmbeding(imgCoverPath,imgEmbedPath,embedZone,factor=0.05):
    '''
    dct嵌入主函数
    '''
    imgCover = cv.imread(imgCoverPath,cv.IMREAD_GRAYSCALE) # 以灰度图方式读取载体图像
    imgEmbed = cv.imread(imgEmbedPath,cv.IMREAD_GRAYSCALE) #以灰度图方式读取嵌入图像
    pixelStream = getEmbedPixel(imgEmbed,factor=factor)
    blockList = divideBlock(imgCover)
    dctBlockList = dctBlock(blockList)
    embedBlock = getRandEmbedBlock(blockList,pixelStream)
    with open('./KeyFile/embedblock.json','w') as fp:
        json.dump(embedBlock,fp)
    rowVlist = []
    pos = 0
    for i in embedBlock:
        pixelList = pixelStream[8*pos:(pos+1)*8]
        pos+=1
        for zone,pixel in zip(embedZone,pixelList):
            # rowVlist.append(float(dctBlockList[i][zone[0]][zone[1]]))
            rowVlist.append(round(float(dctBlockList[i][zone[0]][zone[1]]),7))
            dctBlockList[i][zone[0]][zone[1]] += pixel
    with open('./KeyFile/Vlist.json','w') as fp:
        json.dump(rowVlist,fp)
    idctBlockList = np.uint8(np.rint(idctBlock(dctBlockList)))
    imgStego = mergeBlock(idctBlockList.tolist())
    imgStego = np.array(imgStego,dtype=np.uint8)
    scale = imgEmbedPath[9:].split('.')[0].split('_')[1][4:]
    cv.imwrite('./img/stego{}.bmp'.format(scale),imgStego)
    print('Done!')

# dctEmbeding('./images/lena.bmp','./images/dog_gray.bmp',embedZone,factor=0.06)

if __name__ == "__main__":
    imgCoverPath = input('cover path:')
    imgEmbedPath = input('embed path:')
    factor = float(input('factor:'))
    dctEmbeding(imgCoverPath,imgEmbedPath,embedZone,factor=factor)