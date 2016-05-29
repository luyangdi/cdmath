from pylab import *
from numpy import *
from math import *
#向量内积函数
def vec_inner(v):
    return sum(v * v);

#x为数据，y为观测值
def LM(x,y):
    #初始猜测
    a0=10.0
    b0=0.5
    y_init = zeros(len(x))
    for x_i in range(0,len(x)):
        y_init[x_i] = a0*exp(-b0*x[x_i])
    Ndata = len(y)#数据个数
    Nparams = 2#参数个数
    n_iters = 50#迭代的最大次数
    lamda = 0.01#阻尼系数初值
    #step 1:变量赋值
    updateJ = 1
    a_est = a0
    b_est = b0
    #step 2:迭代
    i = 0
    y_est = zeros(len(x))
    y_est_lm = zeros(len(x))
    while i < n_iters:
        if updateJ == 1:
            #根据当前的值计算雅克比矩阵
            J=zeros([Ndata,Nparams])
            for j in range(0,len(x)):
                J[j][0] = exp(-b_est*x[j])
                J[j][1] = -a_est*x[j]*exp(-b_est*x[j])
            for y_i in range(len(x)):
                y_est[y_i] = a_est*exp(-b_est*x[y_i])#计算当前参数，得到函数值
            d = y - y_est
            H = dot(J.T,J)#计算hessen矩阵
            if i == 0:
                e = vec_inner(d)
        H_lm = H + (lamda*eye(Nparams))
        g = dot(J.T,d)
        dp = dot(inv(H_lm),g)
        a_lm = a_est+dp[0]
        b_lm = b_est+dp[1]
        for y_lm_i in range(len(x)):
            y_est_lm[y_lm_i] = a_lm*exp(-b_lm*x[y_lm_i])
        d_lm = y-y_est_lm
        e_lm = vec_inner(d_lm)
        #print e_lm
        if e_lm < e:
            lamda = lamda/10
            a_est = a_lm
            b_est = b_lm
            e=e_lm
            updateJ = 1
        else:
            #print "update"
            updateJ = 0
            lamda = lamda*10
        i = i + 1
        print "%f\t%f" %(a_est, b_est)
    print "best_parameter_1=%f; best_parameter_2=%f" %(a_est, b_est)

if __name__ == "__main__":
    data_1= [1.2,1.8,2.0,1.5,5.0,3.0,4.0,6.0,8.0]
    obs_1 = [19.21,18.15,15.36,14.10,12.89,9.32,7.45,5.24,3.01]
    LM(data_1,obs_1)
