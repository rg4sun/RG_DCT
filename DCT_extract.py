from cv2 import cv2 as cv 
import numpy as np 
import matplotlib
import random,json

# 每个block的嵌入区，选最中间的中频8个位置
embedZone = [(0,7),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1),(7,0)]

def divideBlock(img):
    '''将图像分割成8x8的块，返回一个list存储了所有的块信息'''
    scale = img.shape[0] # resize过 rowSize 和 columnSize 一样，这里用scale表示
    columnSliceList = [ [column[i*8:(i+1)*8] for i in range(scale//8)] for column in img]
    blockList = []
    for i in range(scale//8):
        rowBlockList = columnSliceList[8*i:8*(i+1)]
        for x in range(scale//8):
            blockList.append([ rowBlockList[y][x] for y in range(8)])
    return blockList


def dctBlock(blockList):
    '''对每一个块进行dct变换'''
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

def pixelStream2Img(pixelStream):
    '''将像素流转换为图像矩阵'''
    scale = int(pow(len(pixelStream),0.5))
    return [ pixelStream[i*scale:(i+1)*scale] for i in range(scale)]


def dctExtracting(imgStegoPath,embedBlock,Vlist,embedZone,factor=0.05):
    '''DCT提取主函数'''
    imgStego = cv.imread(imgStegoPath,cv.IMREAD_UNCHANGED)
    blockList = divideBlock(imgStego)
    dctBlockList = dctBlock(blockList)
    pos = 0
    pixelStream = []
    for i in embedBlock:
        vlistInUsing = Vlist[pos*8:(pos+1)*8]
        pos+=1 # 这里之前没写。。。。
        for zone,v in zip(embedZone,vlistInUsing):
            tmp = dctBlockList[i][zone[0]][zone[1]]
            p = int((tmp - v)/factor)
            pixelStream.append(p)
    imgExtract = pixelStream2Img(pixelStream)
    # print(imgExtract)
    imgExtract = np.array(imgExtract,dtype=np.uint8)
    cv.imwrite('./img/imgExtract.bmp',imgExtract)
    print("Extracting Done!")

if __name__ == "__main__":
    stegoPath = input('stego path:')
    blockPath = input('embedblock file path:')
    vlistPath = input('vlist file path:')
    factor = float(input('factor:'))
    with open(blockPath,'r') as fp:
        embedBlock = json.load(fp)
    with open(vlistPath,'r') as fp:
        Vlist = json.load(fp)
    dctExtracting(stegoPath,embedBlock,Vlist,embedZone,factor=factor)
