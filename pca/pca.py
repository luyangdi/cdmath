# coding=utf-8
'''
PCA
'''
import numpy as np
import os

width=40
height=40

#输入训练图片目录
#trainImgDir=raw_input('enter the directory of train imgs:')

trainImgDir="D:\\python\\pca\\train\\"
trainImgList=os.listdir(trainImgDir)

testImageDir="D:\\python\\pca\\test\\"
testImageList=os.listdir(testImageDir)


grayVectors=[]		#各个灰度向量
avgVector=[]		#平均向量

#输入目录下的各张照片
imgNum=len(trainImgList)
for i in xrange(imgNum):
	print("正在训练:"+trainImgList[i])
	bmp = np.memmap(trainImgDir+trainImgList[i], offset=54, shape=(width,height,3))
	grayImg=bmp[:, :, 2]*0.30+bmp[:, :, 1]*0.59+bmp[:, :, 0]*0.11
	grayVector=[]
	for j in xrange(len(grayImg)):
		grayVector.extend(grayImg[j])
	grayVectors.append(grayVector)

for i in xrange(width*height):
	avgValue=0.0
	for j in xrange(imgNum):
		avgValue+=grayVectors[j][i]
	avgValue=avgValue/imgNum
	avgVector.append(avgValue)

dVectors=[]
for i in xrange(imgNum):
	dVector=np.subtract(grayVectors[i],avgVector)
	dVectors.append(dVector)

#构建协方差矩阵
cMatrix=np.inner(np.transpose(dVectors),np.transpose(dVectors))
cMatrix=cMatrix/imgNum
#print(cMatrix.shape)

U,s,VH=np.linalg.svd(cMatrix)
#print(len(s))
sSum=0.0
sSum2=0.0
p=0
for i in xrange(len(s)):
	sSum+=s[i]
for i in xrange(len(s)):
	sSum2+=s[i]
	p=i+1
	if(sSum2/sSum>0.99):
		break

wMatrix=[]

for i in xrange(p):
	wi=np.inner(cMatrix,np.transpose(VH[i]))/np.sqrt(s[i])
	wMatrix.append(wi)
print(np.array(wMatrix).shape)
#print(np.inner(VH[0],VH[0]))
#print(wMatrix)
#imgNum=len(testImageList)
	
grayVector=[];
#for i in xrange(imgNum):
print("正在测试:"+testImageList[0])
bmp = np.memmap(testImageDir+testImageList[0], offset=54, shape=(width,height,3))
testImage=bmp[:, :, 2]*0.30+bmp[:, :, 1]*0.59+bmp[:, :, 0]*0.11
for j in xrange(len(testImage)):
	grayVector.extend(testImage[j])

tMatrix=[]
for i in xrange(imgNum):
	ti=np.inner(wMatrix,dVectors[i])
	tMatrix.append(ti)
print(np.array(tMatrix).shape)
	
testT=np.inner(wMatrix,np.subtract(grayVector,avgVector))

x=0.0
for i in xrange(imgNum):
	for j in range(i,imgNum):
		xt=np.sqrt(np.dot(np.subtract(tMatrix[i],tMatrix[j]),np.subtract(tMatrix[i],tMatrix[j])))/2
		if(xt>x):
			x=xt


tmp=np.subtract(np.subtract(grayVector,np.inner(np.transpose(wMatrix),testT)),avgVector)

y=np.sqrt(np.dot(tmp,tmp))
print(x)
print(y)
if(y>=x):
	print("不是3")
else:
	print("是3")

