# coding=utf-8

'''
kmeans
'''

import matplotlib.pyplot as plt
import math
import numpy
import random

#dotOringalNum为各个分类最初的大小
dotOringalNum=100
#dotAddNum测试点的个数
dotAddNum=1000

fig = plt.figure()
ax = fig.add_subplot(111)

sets=[]
colors=['g','r','b','y']

#第一个分类，绿色，左下角
a=[]
txx=0.0
tyy=0.0
for i in range(0,dotOringalNum):
	tx=float(random.randint(1000,3000))/100
	ty=float(random.randint(1000,3000))/100
	a.append([tx,ty])
	txx+=tx
	tyy+=ty
#a的第一个元素为a的各个元素xy值之和
a.insert(0,[txx,tyy])
sets.append(a)

#第二个分类，红色，右上角
b=[]
txx=0.0
tyy=0.0
for i in range(0,dotOringalNum):
	tx=float(random.randint(4000,6000))/100
	ty=float(random.randint(4000,6000))/100
	b.append([tx,ty])
	txx+=tx
	tyy+=ty
b.insert(0,[txx,tyy])
sets.append(b)

#第三个分类，蓝色，左上角
c=[]
txx=0.0
tyy=0.0
for i in range(0,dotOringalNum):
	tx=float(random.randint(1000,3000))/100
	ty=float(random.randint(4000,6000))/100
	c.append([tx,ty])
	txx+=tx
	tyy+=ty
c.insert(0,[txx,tyy])
sets.append(c)


#第四个分类，黄色，右下角
d=[]
txx=0
tyy=0
for i in range(0,dotOringalNum):
	tx=float(random.randint(4000,6000))/100
	ty=float(random.randint(1000,3000))/100
	d.append([tx,ty])
	txx+=tx
	tyy+=ty
	#ax.plot([tx],[ty],color=colors[3],linestyle='',marker='.')
d.insert(0,[txx,tyy])
sets.append(d)


#开始测试
for i in range(0,dotAddNum):
	tx=float(random.randint(0,7000))/100
	ty=float(random.randint(0,7000))/100
	dist=9000.0
	setBelong=0
	for j in range(0,4):
		length=len(sets[j])-1

		centX=sets[j][0][0]/length
		centY=sets[j][0][1]/length

		if (centX-tx)*(centX-tx)+(centY-ty)*(centY-ty)<dist:
			setBelong=j
			dist=(centX-tx)*(centX-tx)+(centY-ty)*(centY-ty)
   
	#ax.plot([tx],[ty],color=colors[setBelong],linestyle='',marker='.')
	sets[setBelong][0][0]+=tx
	sets[setBelong][0][1]+=ty
	sets[setBelong].append([tx,ty])

#输出所有点
for i in range(0,4):
	tx=[]
	ty=[]
	for j in range(1,len(sets[i])):
		tx.append(sets[i][j][0])
		ty.append(sets[i][j][1])
	ax.plot(tx,ty,color=colors[i],linestyle='',marker='.')


plt.show()
   