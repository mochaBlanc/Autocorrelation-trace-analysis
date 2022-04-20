


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:55:14 2021

@author: u103424502
"""

import numpy as np
from matplotlib.pylab import plt
from scipy.fftpack import fft,ifft

from temp import plt_acr_data

c=2.99792458


T = 0.01*40/c
Fs = 1/T 
L = 1300

t0=np.arange(L)*T

def plt_cal_data(file):
    data=[[],[]]
    with open(file,'r') as f:
        lines=f.readlines()
        for line in lines:
            ans=line.strip()
            value_float=[float(n) for n in ans.split(' ')]
            data[0].append(value_float[0])
            data[1].append(value_float[1])
    t=np.array(data[0])
    Y=np.array(data[1])
    tt=t/10
    nor=max(Y)-min(Y)
    yy=8*(Y/nor)
    plt.plot(tt,yy)

def cal_data_ifft(file,L_range,left,right,file_adr):
    data=[[],[]]
    with open(file,'r') as f:
        lines=f.readlines()
        for line in lines:
            ans=line.strip()
            value_float=[float(n) for n in ans.split(' ')]
            data[0].append(value_float[0])
            data[1].append(value_float[1])
    t=np.array(data[0])
    Y=np.array(data[1])
    
    Y_fft=fft(Y)
    for ii in range(0,L_range):
        if((ii<left)and(ii>=0)):
            Y_fft[ii]=0
        if((ii<L_range/2)and(ii>=right)):
            Y_fft[ii]=0
        if((ii>=L_range/2)and(ii<L_range-right)):
            Y_fft[ii]=0
        if((ii>=L_range-left)and(ii<L_range)):
            Y_fft[ii]=0
    
    T=0.1 #10*-15
    Fs=1/T
    fre=np.arange(L_range)*Fs
    plt.figure(2)
    plt.plot(fre,Y_fft)
    plt.xlim(0,1000)
    
    plt.figure(1)
    invse_fft=ifft(Y_fft)
    nor=max(invse_fft)-min(invse_fft)
    invse_y=8*(invse_fft/nor)
    plt.plot(t,invse_y,color='darkorange',linewidth=1.5,linestyle='--',label='Cal')
    plt.legend(loc=1,labelspacing=1,frameon=False,fontsize=13)
    
    with open(file_adr,"w") as f:
        for i in range(0,1600):
            str1=str(t[i])+' '+str(invse_y[i])+'\n'
            f.write(str1) 
    
def plt_raw_data(file,x_shift,y_shift,y_scale,x1,x2):
    data=[]
    L_duration=1300
    t=np.arange(L_duration)*T
    with open(file,'r') as f:
        lines=f.readlines()
        for line in lines:
            ans=line.strip().split('\t')
            value_float=list(map(float,ans))
            data.append(value_float[0])
    y=np.array(data)
    nor=max(y)-min(y)
    yy=8*(y/nor)
    plt.plot(t-x_shift,yy*y_scale-y_shift,linewidth=1)
    plt.xlim(x1,x2)

def plt_fft_data(file,x_shift,line_color,linelabel):
    data=[[],[]]
    temp=np.genfromtxt(file,delimiter=' ',dtype='str')
    mapping=np.vectorize(lambda t:complex(t.replace('i','j')))
    p1=mapping(temp)
    for i in range(0,1300):
        data[0].append(p1[i][0])
        data[1].append(p1[i][1])
    x=data[0]
    y=data[1]
    xx=np.array(x)
    yy=np.array(y)
    plt.plot(xx-x_shift,yy,linewidth=2,color=line_color,label=linelabel)
    
        
def plt_ifft_data(file,x_shift,line_color,linelabel,linestyle,linewidth):
    data=[[],[]]
    temp=np.genfromtxt(file,delimiter=' ',dtype='str')
    mapping=np.vectorize(lambda t:complex(t.replace('i','j')))
    p1=mapping(temp)
    for i in range(0,1300):
        data[0].append(p1[i][0])
        data[1].append(p1[i][1])
    x=data[0]
    y=data[1]
    nor=max(y)-min(y)
    yy=8*(y/nor)
    xx=np.array(x)
    plt.xlim(-60,60)
    plt.plot(xx-x_shift,yy,linewidth=linewidth,color=line_color,linestyle=linestyle,label=linelabel)
    plt.xlabel('Delay time (fs)',fontsize=20,labelpad = -0.5)
    plt.ylabel('Intensity (a.u).',fontsize=20,labelpad = -1.5)
    plt.tick_params(labelsize=15)
    plt.legend(loc=1,bbox_to_anchor=(1.04,1.03),labelspacing=0.5,frameon=False,fontsize=15)
    plt.savefig('C:/Users/76385/OneDrive/Desktop/cpr_dimer500_vs_700.png',dpi=500,bbox_inches='tight')

 
file_cal='C:/002/test/acr_18fs_e780_p777_3fs.txt'
file_raw='C:/Users/76385/OneDrive/Desktop/yue/time/dimer/211215/Dimer/Dimer400_26000_1300_4'

file_0='C:/Users/76385/OneDrive/Desktop/yue/time/dimer/211215/exp_cal_cpr/ifft_dimer_400nm_4_26000_1300.txt'
file_1='C:/Users/76385/OneDrive/Desktop/yue/time/dimer/211215/exp_cal_cpr/ifft_dimer_500nm_3_26000_1300.txt'
file_2='C:/Users/76385/OneDrive/Desktop/yue/time/dimer/211215/exp_cal_cpr/ifft_dimer_600nm_2_26000_1300.txt'

file_ifft_dimer500="C:/Users/76385/OneDrive/Desktop/yue/time/dimer/cpr_fin/d500_26000_1300_3/ifft_dimer_500_26000_1300_3.txt"
file_ifft_dimer700="C:/Users/76385/OneDrive/Desktop/yue/time/dimer/cpr_fin/d700_26000_1300_4/ifft_dimer700_26000_1300_4.txt"

file_ifft_block700="C:/Users/76385/OneDrive/Desktop/yue/time/dimer/cpr_fin/block700_26000_1300_1/ifft_block_700_26000_1300_1.txt"
file_ifft_block500="C:/Users/76385/OneDrive/Desktop/yue/time/dimer/cpr_fin/block500_26000_1300_1/ifft_block_500_26000_1300_1.txt"
file_ifft_cal="C:/Users/76385/OneDrive/Desktop/yue/time/dimer/cpr_fin/block700_26000_1300_1/ifft_cal_18fs_e780_p850_2fs.txt"

plt_ifft_data(file_ifft_dimer500,85,'k','Pitch 500 nm','-',2)
plt_ifft_data(file_ifft_dimer700,97,'deepskyblue','Pitch 700 nm','--',1.5)

#plt_ifft_data(file_ifft_cal,0,'darkorange','Fitting','--',1.5)
# cal_data_ifft(file_cal,1600,58,64.5,'C:/Users/76385/OneDrive/Desktop/yue/time/dimer/211215/exp_cal_cpr/cal/ifft_[58,64.5]_cal_acr_E790_18fs_p820_2.5fs.txt')
# plt.figure(1)
# plt.legend(loc=1,labelspacing=0.5,frameon=False,fontsize=12)
# plt.xlabel('Delay time (fs)',fontsize=17)
# plt.ylabel('Intensity a.u.',fontsize=17)
# plt.savefig('C:/Users/76385/OneDrive/Desktop/fit_dimer_pitch600_2.5fs.png',dpi=500,bbox_inches='tight')
#plt_ifft_data(file_2,85,'orange')

# plt_raw_data(file_raw,83,1,1,-50,50)
# plt_cal_data(file_cal)

#plt_fft_data(file_fft_block800,0,'k','Pitch 800')

plt.show()