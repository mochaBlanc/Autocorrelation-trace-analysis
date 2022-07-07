
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:55:14 2021

@author: u103424502
"""

import numpy as np
from matplotlib.pylab import plt
from scipy.fftpack import fft,ifft




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
def raw_data_ifft(file,L_range,left,right,file_adr_fft,file_adr_ifft):
    c=2.99792458
    T = 0.01*40/c
    Fs = 1/T 
    data=[]
    t=np.arange(L_range)*T
    with open(file,'r') as f:
        lines=f.readlines()
        for line in lines:
            ans=line.strip().split('\t')
            value_float=list(map(float,ans))
            data.append(value_float[0])
    y=np.array(data)
    
    y_fft=fft(y)
    for ii in range(0,L_range):
        if((ii<left)and(ii>=0)):
            y_fft[ii]=0
        if((ii<L_range/2)and(ii>=right)):
            y_fft[ii]=0
        if((ii>=L_range/2)and(ii<L_range-right)):
            y_fft[ii]=0
        if((ii>=L_range-left)and(ii<L_range)):
            y_fft[ii]=0
    
    fre=np.arange(L_range)*Fs
    plt.figure(2)
    plt.plot(fre,y_fft)
    plt.xlim(0,1000)
    
     
    plt.figure(1)
    invse_fft=ifft(y_fft)
    nor=max(invse_fft)-min(invse_fft)
    invse_y=8*(invse_fft/nor)
    plt.plot(t,invse_y,color='darkorange',linewidth=1.5,linestyle='--',label='TPI-PL')
    plt.legend(loc=1,labelspacing=1,frameon=False,fontsize=13)
    
    with open(file_adr_fft,"w") as f:
        for i in range(0,L_range):
            f.write(str(fre[i]))
            f.write(" ")
            f.write(str(y_fft[i].real))
            f.write('\n')  
    
    with open(file_adr_ifft,"w") as f:
        for i in range(0,L_range):
            f.write(str(t[i]))
            f.write(" ")
            f.write(str(invse_y[i].real))
            f.write('\n') 
            
def cal_data_ifft(file,L_range,left,right,file_adr_fft,file_adr_ifft):
    data=[[],[]]
    with open(file,'r') as f:
        lines=f.readlines()
        for line in lines:
            ans=line.strip()
            value_float=[float(n) for n in ans.split(' ')]
            data[0].append(value_float[0]/10)
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
    
    with open(file_adr_fft,"w") as f:
        for i in range(0,L_range):
            f.write(str(fre[i]))
            f.write(" ")
            f.write(str(Y_fft[i].real))
            f.write('\n')  
    
    with open(file_adr_ifft,"w") as f:
        for i in range(0,L_range):
            f.write(str(t[i]))
            f.write(" ")
            f.write(str(invse_y[i].real))
            f.write('\n') 
    
def plt_raw_data(file,x_shift,y_shift,y_scale,x1,x2):
    data=[]
    L_duration=1300
    c=2.99792458
    T = 0.01*40/c
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

def plt_fft_data(file,L_range,line_color,linelabel):
    data=[[],[]]
    temp=np.genfromtxt(file,delimiter=' ',dtype='str')
    mapping=np.vectorize(lambda t:float(t.replace('i','j')))
    p1=mapping(temp)
    for i in range(0,L_range):
        data[0].append(p1[i][0])
        data[1].append(p1[i][1])
    x=data[0]
    y=data[1]
    xx=np.array(x)
    yy=np.array(y)
    
    plt.figure(2)
    plt.plot(xx*0.1,yy,color=line_color,label=linelabel)
    plt.xlabel('Frequency (THz)',fontsize=20,labelpad = -0.5)
    plt.ylabel('Amplitude',fontsize=20,labelpad = 1)
    plt.yticks([])
    plt.tick_params(labelsize=15)
    plt.xlim((0,200))
    plt.legend(loc=1,bbox_to_anchor=(1.01,1.03),labelspacing=0.5,frameon=False,fontsize=15)
    #plt.savefig('C:/Users/76385/OneDrive/Desktop/fft_block_pitch500.png',dpi=500,bbox_inches='tight')
    plt.show()
    
        
def plt_ifft_data(file,L_range,x_shift,line_color,linelabel,linestyle,linewidth):
    data=[[],[]]
    temp=np.genfromtxt(file,delimiter=' ',dtype='str')
    mapping=np.vectorize(lambda t:float(t.replace('i','j')))
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
    #plt.savefig('C:/Users/76385/OneDrive/Desktop/cpr_dimer500_vs_700.png',dpi=500,bbox_inches='tight')

 
file_cal='C:/Users/1000297123/Desktop/yue/time/dimer/cpr_fin/block300_26000_1300_1/acr_18fs_e780_p795_2fs.txt'
file_raw='C:/Users/1000297123/Desktop/yue/time/dimer/cpr_fin/block300_26000_1300_1/block300_26000_1300_A_1'

file_ifft_out='C:/Users/1000297123/Desktop/ifft_block300_26000_1300_A_1.txt'
file_fft_out='C:/Users/1000297123/Desktop/fft_block300_26000_1300_A_1.txt'
cal_fft_out='C:/Users/1000297123/Desktop/cal_fft__18fs_e780_p795_2fs.txt'
cal_ifft_out='C:/Users/1000297123/Desktop/cal_ifft__18fs_e780_p795_2fs.txt'




#plt_cal_data(file_cal)
#plt_raw_data(file_raw,107,1.5,1,-50,50)

#raw_data_ifft(file_raw,1300,50,80,file_fft_out,file_ifft_out)
#cal_data_ifft(file_cal,1601,56,75,cal_fft_out,cal_ifft_out)

plt_ifft_data(file_ifft_out,1300,109,'k','Pitch 300 nm','-',2)
#plt_fft_data(file_fft_out,1300,'k','Block 300')

plt_ifft_data(cal_ifft_out,1601,0,'darkorange','Cal 300 nm','-',2)
# plt_fft_data(cal_fft_out,1601,'k','cal')


plt.show()
