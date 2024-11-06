import matplotlib.pyplot as plt
from LMDW import *
from RRDW1 import *
from RRDW2 import *
from URRDW import *
from URFMGRR import *
import numpy as np
from preprocess import *
import diffprivlib
import pandas as pd
from matplotlib.font_manager import FontProperties# 步骤1
from brokenaxes import brokenaxes
import os
import os

import matplotlib
# matplotlib.use("TkAgg")
# matplotlib.rcParams['font.family'] = 'Times New Roman'
# # 使字体同非 mathtext 字的字体，此处即 Times New Roman
# matplotlib.rcParams['mathtext.default'] = 'regular'
# import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from matplotlib import rcParams
#plt.rc('text', usetex=True)
# plt.rc('text', usetex=True)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

config = {
        "font.family": "Times New Roman",
        "font.size": 20,
        "mathtext.fontset": 'cm', # 'cm'区分“\varepsilon”看起来不同于“\epsilon”
        "font.serif": ['SimSun'],
    }
rcParams.update(config)

#"mathtext.fontset": 'stix'

np.random.seed(1)
Ks = generate_random_key()# generate secret key
watermark=[1 for _ in range(32)]# generate watermark with length 32
FileType=['pdf', 'svg']
SaveFileType=FileType[1]
epsilon = 1
p2=0.5
power=0
delta=0
sen_ratio=0.5
#----------------analyse statistical distortion of perturbed database-----------------#
select=2
def select_dataset(select):
    datatype=['Normal distribution','Uniform distribution','Adult','Drug','PhiUSIIL Phishing','Heart disease','MI']
    type=datatype[select]
    if select ==0:
        df=generate_normal_data(10000)
    elif select==1:
        df=generate_synthetic_data('uniform',10000)
    elif select > 1:
        df=load_database(datatype[select])# df is original database
    array_data = np.array(df)# datafram to array
    return df,array_data,type
df,array_data,datatype=select_dataset(select)

print(max(df[:,1]))
print(min(df[:,1]))
print(max(df[:,2]))
print(min(df[:,2]))

#get_one_attributte_frequency_distribution(df[:,2])

# sen_ratio lower than 0.8
def compute_statical_distortion(epsilon,df,watermark,Ks):
    array_data = np.array(df)# to array
    original_age_mean = np.mean(array_data[:,1])
    original_age_std = np.std(array_data[:,1])
    original_workinghours_mean = np.mean(array_data[:,2])
    original_workinghours_std = np.std(array_data[:,2])

    print("original database distortion")
    print(original_age_mean,end=' ')
    print(original_age_std,end='\n')
    print(original_workinghours_mean,end=' ')
    print(original_workinghours_std,end='\n')

    embed_method=[embeddedLMDW,embeddedRRDW1,embeddedRRDW2,URRDW,URFMGRR]

    for method in embed_method:
        dw_age_mean=[]
        dw_age_std=[]
        dw_workinghours_mean=[]
        dw_workinghours_std=[]
        for _ in range(10):
            if method==URFMGRR:
                dw_perturbed_database, mark = method(epsilon, delta, df,sen_ratio,p2, watermark, Ks)
            elif method==URRDW:
                dw_perturbed_database, mark=method(epsilon, df, sen_ratio, watermark, Ks)
            else:
                dw_perturbed_database, mark = method(epsilon, df, watermark, Ks)

            array_dw =np.array(dw_perturbed_database)
            dw_age_mean.append(np.mean(array_dw[:,1]))
            dw_age_std.append(np.std(array_dw[:,1]))
            dw_workinghours_mean.append(np.mean(array_dw[:,2]))
            dw_workinghours_std.append(np.std(array_dw[:,2]))

        print("%s distortion" % method)
        print(np.mean(dw_age_mean),end=' ')
        print(np.mean(dw_age_std),end ='\n')
        print(np.mean(dw_workinghours_mean),end=' ')
        print(np.mean(dw_workinghours_std), end='\n')

#compute_statical_distortion(epsilon,df,watermark,Ks)

def compute_DM_DS(epsilon,df,watermark,Ks):

    embed_method = [embeddedLMDW, embeddedRRDW1, embeddedRRDW2, URRDW, URFMGRR]

    for method in embed_method:

        array_data = np.array(df)  # to array
        ori_age_mean = np.mean(array_data[:, 1])
        ori_age_std = np.std(array_data[:, 1])
        ori_workinghours_mean = np.mean(array_data[:, 2])
        ori_workinghours_std = np.std(array_data[:, 2])

        result_age = [[] for _ in range(2)]
        result_workinghours=[[] for _ in  range(2)]
        Mse=[]
        for _ in range(10):
            if method==URFMGRR:
                dw_perturbed_database, mark = method(epsilon,delta, df,sen_ratio,p2, watermark, Ks)
            elif method==URRDW:
                dw_perturbed_database, mark = method(epsilon, df, sen_ratio, watermark, Ks)
            else:
                dw_perturbed_database, mark = method(epsilon, df, watermark, Ks)
            array_dw = np.array(dw_perturbed_database)
            dw_age_mean = np.mean(array_dw[:, 1])
            dw_age_std = np.std(array_dw[:, 1])
            dw_workinghours_mean = np.mean(array_dw[:, 2])
            dw_workinghours_std = np.std(array_dw[:, 2])

            result_age[0].append(abs(dw_age_mean-ori_age_mean))
            result_age[1].append(abs(dw_age_std-ori_age_std))
            result_workinghours[0].append(abs(dw_workinghours_mean-ori_workinghours_mean))
            result_workinghours[1].append(abs(dw_workinghours_std-ori_workinghours_std))

            Mse.append(mean_absolute_error(df, dw_perturbed_database))

        ageMean_var=np.mean(result_age[0])
        ageStd_var = np.mean(result_age[1])

        workinghoursMean_var=np.mean(result_workinghours[0])
        workinghoursStd_var = np.mean(result_workinghours[1])

        Mse_var=np.mean(Mse)
        print("%s distortion" % method)
        print(ageMean_var, end=' ')
        print(ageStd_var, end='\n')
        print(workinghoursMean_var,end=' ')
        print(workinghoursStd_var,end='\n')
        print(Mse_var)

#compute_DM_DS(epsilon,df,watermark,Ks)

#-----------------------------analyse MSE between original database and perturbed one ---------------------------------#
def figureshow_diff_epsilon(df,watermark,Ks):

    epsilons = [0.2,0.4,0.6,0.8,1.0]
    # index = np.arange(len(epsilons))
    font_options = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    lmdw=[]
    rrdw1=[]
    rrdw2=[]
    urrdw=[]
    urfmgrrdw=[]
    for epsilon in epsilons:
        lmdw_perturb_database, mark = embeddedLMDW(epsilon, df,watermark,Ks)
        rrdw1_perturb_database, mark = embeddedRRDW1(epsilon,df, watermark,Ks)
        rrdw2_perturb_database, mark = embeddedRRDW2(epsilon,df, watermark,Ks)
        urrdw_perturb_database, mark = URRDW(epsilon, df, sen_ratio,watermark,Ks)
        urfmgrrdw_perturb_database, mark = URFMGRR(epsilon, delta, df, sen_ratio,p2,watermark, Ks)

        lmdw.append(mean_squared_error(df, lmdw_perturb_database))
        rrdw1.append(mean_squared_error(df, rrdw1_perturb_database))
        rrdw2.append(mean_squared_error(df, rrdw2_perturb_database))
        urrdw.append(mean_squared_error(df, urrdw_perturb_database))
        urfmgrrdw.append(mean_squared_error(df, urfmgrrdw_perturb_database))

    dw_list=[lmdw,rrdw1,rrdw2,urrdw,urfmgrrdw]


    lmdw=dw_list[0]
    rrdw1=dw_list[1]
    rrdw2=dw_list[2]
    urrdw=dw_list[3]
    urfmgrrdw=dw_list[4]
    fig, axes = plt.subplots(figsize=(10, 6),dpi=100)  # 创建一个图形对象和一个子图对象

    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    xAxisfont = {'family': 'Times New Roman', 'size': 20,'style': 'italic'}  # 你可以调整size的值来改变字体大小
    xyAxisFont = FontProperties(fname=r'./times.ttf', size=20)  # 步骤2
    legendFont = FontProperties(fname=r'./times.ttf', size=12)

    axes.plot(epsilons, lmdw,'x',lw=1,color='green',linestyle='-',label='LMDW')
    axes.plot(epsilons, rrdw1,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    axes.plot(epsilons, rrdw2,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    axes.plot(epsilons, urfmgrrdw,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')


    axes.set_xlabel(r'$\epsilon$',fontproperties=xAxisfont)
    axes.set_ylabel(r'$MSE$ of sensitive attribute values',fontproperties=xAxisfont)
    if delta ==0:
        axes.set_title(r'%s ($\delta=%d$, $p=%.1f$, ratio=%.1f)' % (datatype, delta,p2,sen_ratio), fontproperties=xyAxisFont)
    else:
        axes.set_title(r'%s ($\delta=10^{%d}$, $p=%.1f$, ratio=%.1f)' % (datatype,power,p2,sen_ratio),fontproperties=xyAxisFont)
    axes.legend(prop=legendfont)
    axes.grid()#生成网格

    plt.xlim(0.2,1)
    x1_label = axes.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = axes.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    axes.tick_params(axis='y',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    axes.tick_params(axis='x',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    if SaveFileType=='pdf':
        filename='figures/mse_diff_epsilon_%s.pdf' % datatype
    elif SaveFileType=='svg':
        filename = 'figures/mse_diff_epsilon_%s.svg' % datatype
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()
figureshow_diff_epsilon(df,watermark,Ks)

def figureshow_diff_delta(df,watermark,Ks):

    #deltas=[i*0.000001 for i in range(10)]
    deltas = [pow(10,-i) for i in range(4,9)]
    # index = np.arange(len(epsilons))
    font_options = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    lmdw=[]
    rrdw1=[]
    rrdw2=[]
    urrdw=[]
    urfmgrrdw=[]
    for delta in deltas:
        lmdw_perturb_database, mark = embeddedLMDW(epsilon, df,watermark,Ks)
        rrdw1_perturb_database, mark = embeddedRRDW1(epsilon,df, watermark,Ks)
        rrdw2_perturb_database, mark = embeddedRRDW2(epsilon,df, watermark,Ks)
        urrdw_perturb_database, mark = URRDW(epsilon, df, sen_ratio,watermark,Ks)
        urfmgrrdw_perturb_database, mark = URFMGRR(epsilon, delta, df, sen_ratio,p2,watermark, Ks)

        lmdw.append(mean_squared_error(df, lmdw_perturb_database))
        rrdw1.append(mean_squared_error(df, rrdw1_perturb_database))
        rrdw2.append(mean_squared_error(df, rrdw2_perturb_database))
        urrdw.append(mean_squared_error(df, urrdw_perturb_database))
        urfmgrrdw.append(mean_squared_error(df, urfmgrrdw_perturb_database))

    dw_list=[lmdw,rrdw1,rrdw2,urrdw,urfmgrrdw]


    lmdw=dw_list[0]
    rrdw1=dw_list[1]
    rrdw2=dw_list[2]
    urrdw=dw_list[3]
    urfmgrrdw=dw_list[4]
    fig, axes = plt.subplots(figsize=(10, 6),dpi=200)  # 创建一个图形对象和一个子图对象

    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    xAxisfont = {'family': 'Times New Roman', 'size': 20,'style': 'italic'}  # 你可以调整size的值来改变字体大小
    xyAxisFont = FontProperties(fname=r'./times.ttf', size=20)  # 步骤2
    legendFont = FontProperties(fname=r'./times.ttf', size=12)

    axes.plot(deltas, lmdw,'x',lw=1,color='green',linestyle='-',label='LMDW')
    axes.plot(deltas, rrdw1,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    axes.plot(deltas, rrdw2,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    axes.plot(deltas, urfmgrrdw,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')
    plt.xscale('log')

    axes.set_xlabel(r'$\delta$',fontproperties=xAxisfont)
    # axes.set_ylabel(r'$\textit{MSE}$',fontproperties=xAxisfont)
    axes.set_ylabel(r'$MSE$ of sensitive attribute values',fontproperties=xAxisfont)
    # axes.set_ylabel('中文',fontproperties=xAxisfont)
    axes.set_title(r'%s ($\epsilon=%d$, $p_2=%.1f$, ratio=%.2f)' % (datatype,epsilon,p2,sen_ratio),fontproperties=xyAxisFont)
    axes.legend(prop=legendfont)
    axes.grid()#生成网格

    #plt.xlim(0.2,0.9)
    x1_label = axes.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = axes.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]



    axes.tick_params(axis='y',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    axes.tick_params(axis='x',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    if SaveFileType=='pdf':
        filename='figures/mse_diff_delta_%s.pdf' % datatype
    elif SaveFileType=='svg':
        filename = 'figures/mse_diff_delta_%s.svg' % datatype
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()
#figureshow_diff_delta(df,watermark,Ks)


def figureshow_diff_p2(df,watermark,Ks):

    p2_list = [i*0.1 for i in range(1,10)]
    # index = np.arange(len(epsilons))
    font_options = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    lmdw=[]
    rrdw1=[]
    rrdw2=[]
    urrdw=[]
    urfmgrrdw=[]
    for p2 in p2_list:
        lmdw_perturb_database, mark = embeddedLMDW(epsilon, df,watermark,Ks)
        rrdw1_perturb_database, mark = embeddedRRDW1(epsilon,df, watermark,Ks)
        rrdw2_perturb_database, mark = embeddedRRDW2(epsilon,df, watermark,Ks)
        #urrdw_perturb_database, mark = URRDW(epsilon, df, sen_ratio,watermark,Ks)
        urfmgrrdw_perturb_database, mark = URFMGRR(epsilon, delta, df, sen_ratio,p2,watermark, Ks)

        lmdw.append(mean_squared_error(df, lmdw_perturb_database))
        rrdw1.append(mean_squared_error(df, rrdw1_perturb_database))
        rrdw2.append(mean_squared_error(df, rrdw2_perturb_database))
        #urrdw.append(mean_squared_error(df, urrdw_perturb_database))
        urfmgrrdw.append(mean_squared_error(df, urfmgrrdw_perturb_database))

    dw_list=[lmdw,rrdw1,rrdw2,urfmgrrdw]

    fig, axes = plt.subplots(figsize=(10, 6),dpi=200)  # 创建一个图形对象和一个子图对象

    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    xAxisfont = {'family': 'Times New Roman', 'size': 20,'style': 'italic'}  # 你可以调整size的值来改变字体大小
    xyAxisFont = FontProperties(fname=r'./times.ttf', size=20)  # 步骤2
    legendFont = FontProperties(fname=r'./times.ttf', size=12)

    axes.plot(p2_list, lmdw,'x',lw=1,color='green',linestyle='-',label='LMDW')
    axes.plot(p2_list, rrdw1,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    axes.plot(p2_list, rrdw2,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    axes.plot(p2_list, urfmgrrdw,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')
    #plt.xscale('log')

    axes.set_xlabel(r'$p$',fontproperties=xAxisfont)
    # axes.set_ylabel(r'$\textit{MSE}$',fontproperties=xAxisfont)
    axes.set_ylabel(r'$MSE$ of sensitive attribute values',fontproperties=xAxisfont)
    # axes.set_ylabel('中文',fontproperties=xAxisfont)
    if delta==0:
        axes.set_title(r'%s ($\epsilon=%.1f$, $\delta=%d$, ratio=%.1f)' % (datatype,epsilon,delta,sen_ratio),fontproperties=xyAxisFont)
    axes.legend(prop=legendfont)
    axes.grid()#生成网格

    plt.xlim(0.1,0.9)
    x1_label = axes.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = axes.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]


    axes.tick_params(axis='y',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    axes.tick_params(axis='x',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    if SaveFileType=='pdf':
        filename='figures/mse_diff_p2_%s.pdf' % datatype
    elif SaveFileType=='svg':
        filename = 'figures/mse_diff_p2_%s.svg' % datatype
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()
#figureshow_diff_p2(df,watermark,Ks)


#--------------------------------analyse  frequency estimation : age, workinghour--------------------------------#
def distribution_estimation(epsilon,df,watermark,Ks):
    array_data=np.array(df)
    max_age = np.max(array_data[:,1])
    min_age = np.min(array_data[:,1])
    min_workinghour = np.min(array_data[:,2])
    max_workinghour = np.max(array_data[:,2])
    age_points = np.linspace(min_age,max_age,10)
    workinghour_points = np.linspace(min_workinghour,max_workinghour,10)
    age_points = np.round(age_points).astype(int)
    workinghour_points = np.round(workinghour_points).astype(int)

    area_original = array_data[:, 1]

    lmdw_perturbed_database,mark = embeddedLMDW(epsilon,df,watermark,Ks)
    array_lmdw =np.array(lmdw_perturbed_database)
    area_lmdw = array_lmdw[:,1]

    rrdw1_perturbed_database,mark = embeddedRRDW1(epsilon,df,watermark,Ks)
    array_rrdw1 = np.array(rrdw1_perturbed_database)
    area_rrdw1 = array_rrdw1[:,1]

    rrdw2_perturbed_database,mark = embeddedRRDW2(epsilon,df,watermark,Ks)
    array_rrdw2 = np.array(rrdw2_perturbed_database)
    area_rrdw2 = array_rrdw2[:,1]

    urrdw_perturbed_database, mark = URRDW(epsilon, df,sen_ratio, watermark, Ks)
    array_urrdw = np.array(urrdw_perturbed_database)
    area_urrdw = array_urrdw[:,1]

    urfmgrrdw_perturbed_database, mark = URFMGRR(epsilon, delta, df,sen_ratio, watermark, Ks)#proposed scheme
    array_urfmgrrdw = np.array(urfmgrrdw_perturbed_database)
    area_urfmgrrdw = array_urfmgrrdw[:,1]

    #ticker_spacing=2
    #age attribute
    edges = np.array(age_points)

    counts = np.zeros((len(age_points)-1,6))
    dbs = [area_original,area_lmdw,area_rrdw1,area_rrdw2,area_urrdw,area_urfmgrrdw]
    for i, db in enumerate(dbs):
        counts[:, i], _ = np.histogram(db, bins=edges)
    print(counts)


    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    font = {'family': 'Times New Roman', 'size': 16}  # 你可以调整size的值来改变字体大小
    xyAxisfont = {'family': 'Times New Roman', 'size': 20}  # 你可以调整size的值来改变字体大小
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(len(edges)-1)
    bar_width = 0.15/7*5


    ax.bar(index + 0 * bar_width, counts[:, 0], bar_width, label='Original database')
    ax.bar(index + 1 * bar_width, counts[:, 1], bar_width, label='LMDW')
    ax.bar(index + 2 * bar_width, counts[:, 2], bar_width, label='RRDW1')
    ax.bar(index + 3 * bar_width, counts[:, 3], bar_width, label='RRDW2')
    #ax.bar(index + 4 * bar_width, counts[:, 4], bar_width, label='URRDW')
    ax.bar(index + 5 * bar_width, counts[:, 5], bar_width, label='URFMGRRDW')

    # 设置图表标签和标题
    ax.set_xlabel('Attribute 1',fontdict=xyAxisfont)
    ax.set_ylabel('Estimation of the distribution',fontdict=xyAxisfont)
    ax.set_title(r'Four watermarking methods ($\epsilon=1$, $\delta=0.5$)',fontdict=xyAxisfont)
    ax.set_xticks(index + bar_width * (counts.shape[1]-1) / 2)
    ax.set_xticklabels([f'{edges[j]}-{edges[j+1]}' for j in range(len(edges)-1)], fontdict=font)
    ax.legend(prop=legendfont)
    y_labels = ax.get_yticks()  # 获取当前的y轴刻度位置
    ax.set_yticklabels(['{:.0f}'.format(y) for y in y_labels], fontdict=font)
    #ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ticker_spacing))
    plt.xticks(rotation=-15)
    plt.tight_layout()
    filename = 'figures/ages_distribution_unif.pdf'
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()


    #workinghour attribute
    edges = np.array(workinghour_points)

    area_original = array_data[:,2]
    area_lmdw = array_lmdw[:,2]
    area_rrdw1 = array_rrdw1[:,2]
    area_rrdw2 = array_rrdw2[:,2]
    area_urrdw = array_urrdw[:,2]
    area_urfmgrrdw = array_urfmgrrdw[:,2]

    counts = np.zeros((len(workinghour_points)-1,6))
    dbs = [area_original,area_lmdw,area_rrdw1,area_rrdw2,area_urrdw,area_urfmgrrdw]
    for i, db in enumerate(dbs):
        counts[:, i], _ = np.histogram(db, bins=edges)
    print(counts)

    fig, ax = plt.subplots(figsize=(12, 8))
    index = np.arange(len(edges)-1)
    bar_width = 0.15/7*5

    ax.bar(index + 0 * bar_width, counts[:, 0], bar_width, label='Original database')
    ax.bar(index + 1 * bar_width, counts[:, 1], bar_width, label='LMDW')
    ax.bar(index + 2 * bar_width, counts[:, 2], bar_width, label='RRDW1')
    ax.bar(index + 3 * bar_width, counts[:, 3], bar_width, label='RRDW2')
    #ax.bar(index + 4 * bar_width, counts[:, 4], bar_width, label='URRDW')
    ax.bar(index + 5 * bar_width, counts[:, 5], bar_width, label='URFMGRRDW')

    # 设置图表标签和标题
    ax.set_xlabel('Attribute 2',fontdict=xyAxisfont)
    ax.set_ylabel('Estimation of the distribution',fontdict=xyAxisfont)
    ax.set_title(r'Four watermarking methods ($\epsilon=1$, $\delta=0.5$)',fontdict=xyAxisfont)
    ax.set_xticks(index + bar_width * (counts.shape[1]-1) / 2)
    ax.set_xticklabels([f'{edges[j]}-{edges[j+1]}' for j in range(len(edges)-1)], fontdict=font)
    ax.legend(prop=legendfont)
    y_labels = ax.get_yticks()  # 获取当前的y轴刻度位置
    ax.set_yticklabels(['{:.0f}'.format(y) for y in y_labels], fontdict=font)

    plt.xticks(rotation=-15)
    plt.tight_layout()
    filename = 'figures/workinghour_distribution_unif.pdf'
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
#distribution_estimation(epsilon,df,watermark,Ks)

def distribution_estimation_mse_diff_epsilon(df,watermark,Ks):
    array_data=np.array(df)
    max_age = np.max(array_data[:,1])
    min_age = np.min(array_data[:,1])
    min_workinghour = np.min(array_data[:,2])
    max_workinghour = np.max(array_data[:,2])
    age_points = np.linspace(min_age,max_age,10)
    workinghour_points = np.linspace(min_workinghour,max_workinghour,10)
    age_points = np.round(age_points).astype(int)
    workinghour_points = np.round(workinghour_points).astype(int)

    area_original1 = array_data[:, 1]
    area_original2 = array_data[:, 2]

    # epsilons_0_1 = [0.1,0.2, 0.4, 0.6, 0.8]
    # epsilons_1_5 = [1*i for i in range(1,6)]
    # epsilons=[epsilons_0_1,epsilons_1_5]
    epsilons= [0.1, 0.2, 0.4, 0.6, 0.8,1,2,3,4,5]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lmdw_mse1=[]
    rrdw1_mse1=[]
    rrdw2_mse1=[]
    urrdw_mse1=[]
    urfmgrrdw_mse1=[]

    lmdw_mse2=[]
    rrdw1_mse2=[]
    rrdw2_mse2=[]
    urrdw_mse2=[]
    urfmgrrdw_mse2=[]

    for epsilon in epsilons:

        lmdw_perturbed_database, mark = embeddedLMDW(epsilon, df,watermark,Ks)
        rrdw1_perturbed_database, mark = embeddedRRDW1(epsilon,df, watermark,Ks)
        rrdw2_perturbed_database, mark = embeddedRRDW2(epsilon,df, watermark,Ks)
        #urrdw_perturbed_database, mark = URRDW(epsilon, df, sen_ratio,watermark,Ks)
        urfmgrrdw_perturbed_database, mark = URFMGRR(epsilon, delta, df, sen_ratio,p2,watermark, Ks)

        array_lmdw =np.array(lmdw_perturbed_database)
        area_lmdw = array_lmdw[:,1]

        array_rrdw1 = np.array(rrdw1_perturbed_database)
        area_rrdw1 = array_rrdw1[:,1]

        array_rrdw2 = np.array(rrdw2_perturbed_database)
        area_rrdw2 = array_rrdw2[:,1]

        #array_urrdw = np.array(urrdw_perturbed_database)
        #area_urrdw = array_urrdw[:,1]

        array_urfmgrrdw = np.array(urfmgrrdw_perturbed_database)
        area_urfmgrrdw = array_urfmgrrdw[:,1]

        lmdw_mse1.append(get_frequency_distribution(area_original1, area_lmdw))
        rrdw1_mse1.append(get_frequency_distribution(area_original1, area_rrdw1))
        rrdw2_mse1.append(get_frequency_distribution(area_original1, area_rrdw2))
        urfmgrrdw_mse1.append(get_frequency_distribution(area_original1, area_urfmgrrdw))

        lmdw_mse2.append(get_frequency_distribution(area_original2, array_lmdw[:,2]))
        rrdw1_mse2.append( get_frequency_distribution(area_original2, array_rrdw1[:,2]))
        rrdw2_mse2.append( get_frequency_distribution(area_original2, array_rrdw2[:,2]))
        urfmgrrdw_mse2.append(get_frequency_distribution(area_original2,  array_urfmgrrdw[:,2]))

    # display
    # attribute 1
    fig, axes = plt.subplots(figsize=(10, 6),dpi=100)  # 创建一个图形对象和一个子图对象

    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    xAxisfont = {'family': 'Times New Roman', 'size': 20,'style': 'italic'}  # 你可以调整size的值来改变字体大小
    xyAxisFont = FontProperties(fname=r'./times.ttf', size=20)  # 步骤2
    legendFont = FontProperties(fname=r'./times.ttf', size=12)

    axes.plot(x, lmdw_mse1,'x',lw=1,color='green',linestyle='-',label='LMDW')
    axes.plot(x, rrdw1_mse1,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    axes.plot(x, rrdw2_mse1,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    axes.plot(x, urfmgrrdw_mse1,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')
    plt.yscale('log')

    axes.set_xlabel(r'$\epsilon$',fontproperties=xyAxisFont)
    axes.set_ylabel(r'$log$ (MSE) of frequency estimation',fontproperties=xyAxisFont)
    if delta==0:
        axes.set_title(r'Attribute 1 of %s ($\delta=%d$, $p$=%.1f, ratio=%.1f)' % (datatype, delta,p2,sen_ratio), fontproperties=xyAxisFont)
    else:
        axes.set_title(r'Attribute 1 of %s ($\delta=10^{%d}$, $p$=%.1f, ratio=%.1f)' % (datatype,power,p2,sen_ratio),fontproperties=xyAxisFont)
    axes.legend(prop=legendfont)
    axes.grid()#生成网格

    plt.xlim(0.95,10.1)
    #plt.xlim(1, 10)
    # x1_label = axes.get_xticklabels()
    # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    # y1_label = axes.get_yticklabels()
    # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    #plt.xticks(fontproperties='Times New Roman', fontsize=16)
    #axes.set_xticks(epsilons,fontproperties='Times New Roman', fontsize=16)
    x = [1, 2, 3, 4, 5, 6,7,8,9,10]
    _ = plt.xticks(x,epsilons,fontproperties='Times New Roman', fontsize=20)


    # axes.tick_params(axis='y',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    # axes.tick_params(axis='x',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    if SaveFileType == 'pdf':
        filename='figures/%s/att1_frequency_mse_epsilon0_1_%s.pdf' % (datatype,datatype)
    elif SaveFileType=='svg':
        filename = 'figures/%s/att1_frequency_mse_epsilon0_1_%s.svg' % (datatype,datatype)
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()

    # attribute 2
    fig, axes = plt.subplots(figsize=(10, 6),dpi=100)  # 创建一个图形对象和一个子图对象

    axes.plot(x, lmdw_mse2,'x',lw=1,color='green',linestyle='-',label='LMDW')
    axes.plot(x, rrdw1_mse2,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    axes.plot(x, rrdw2_mse2,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    axes.plot(x, urfmgrrdw_mse2,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')
    plt.yscale('log')

    axes.set_xlabel(r'$\epsilon$',fontproperties=xyAxisFont)
    axes.set_ylabel(r'$log$ (MSE) of frequency estimation',fontproperties=xyAxisFont)
    if delta==0:
        axes.set_title(r'Attribute 2 of %s ($\delta=%d$, $p$=%.1f, ratio=%.1f)' % (datatype, delta,p2,sen_ratio), fontproperties=xyAxisFont)
    else:
        axes.set_title(r'Attribute 2 of %s ($\delta=10^{%d}$, $p$=%.1f, ratio=%.1f)' % (datatype,power,p2,sen_ratio),fontproperties=xyAxisFont)
    axes.legend(prop=legendfont)
    axes.grid()#生成网格

    plt.xlim(0.95,10.1)
    _ = plt.xticks(x, epsilons, fontproperties='Times New Roman', fontsize=20)
    #plt.xlim(1, 10)
    # x1_label = axes.get_xticklabels()
    # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    # y1_label = axes.get_yticklabels()
    # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    #
    #
    # axes.tick_params(axis='y',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    # axes.tick_params(axis='x',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    if SaveFileType == 'pdf':
        filename='figures/%s/att2_frequency_mse_epsilon0_1_%s.pdf' % (datatype,datatype)
    elif SaveFileType == 'svg':
        filename = 'figures/%s/att2_frequency_mse_epsilon0_1_%s.svg' % (datatype,datatype)
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()

distribution_estimation_mse_diff_epsilon(df,watermark,Ks)

def distribution_estimation_logmse_diff_epsilon(df,watermark,Ks):
    array_data=np.array(df)
    max_age = np.max(array_data[:,1])
    min_age = np.min(array_data[:,1])
    min_workinghour = np.min(array_data[:,2])
    max_workinghour = np.max(array_data[:,2])
    age_points = np.linspace(min_age,max_age,10)
    workinghour_points = np.linspace(min_workinghour,max_workinghour,10)
    age_points = np.round(age_points).astype(int)
    workinghour_points = np.round(workinghour_points).astype(int)

    area_original1 = array_data[:, 1]
    area_original2 = array_data[:, 2]

    epsilons = [0.2, 0.4, 0.6, 0.8, 1.0]
    #epsilons = [1*i for i in range(1,11)]
    lmdw_mse1=[]
    rrdw1_mse1=[]
    rrdw2_mse1=[]
    urrdw_mse1=[]
    urfmgrrdw_mse1=[]

    lmdw_mse2=[]
    rrdw1_mse2=[]
    rrdw2_mse2=[]
    urrdw_mse2=[]
    urfmgrrdw_mse2=[]

    for epsilon in epsilons:

        lmdw_perturbed_database, mark = embeddedLMDW(epsilon, df,watermark,Ks)
        rrdw1_perturbed_database, mark = embeddedRRDW1(epsilon,df, watermark,Ks)
        rrdw2_perturbed_database, mark = embeddedRRDW2(epsilon,df, watermark,Ks)
        #urrdw_perturbed_database, mark = URRDW(epsilon, df, sen_ratio,watermark,Ks)
        urfmgrrdw_perturbed_database, mark = URFMGRR(epsilon, delta, df, sen_ratio,p2,watermark, Ks)

        array_lmdw =np.array(lmdw_perturbed_database)
        area_lmdw = array_lmdw[:,1]

        array_rrdw1 = np.array(rrdw1_perturbed_database)
        area_rrdw1 = array_rrdw1[:,1]

        array_rrdw2 = np.array(rrdw2_perturbed_database)
        area_rrdw2 = array_rrdw2[:,1]

        #array_urrdw = np.array(urrdw_perturbed_database)
        #area_urrdw = array_urrdw[:,1]

        array_urfmgrrdw = np.array(urfmgrrdw_perturbed_database)
        area_urfmgrrdw = array_urfmgrrdw[:,1]

        lmdw_mse1.append(get_frequency_distribution(area_original1, area_lmdw))
        rrdw1_mse1.append(get_frequency_distribution(area_original1, area_rrdw1))
        rrdw2_mse1.append(get_frequency_distribution(area_original1, area_rrdw2))
        urfmgrrdw_mse1.append(get_frequency_distribution(area_original1, area_urfmgrrdw))

        lmdw_mse2.append(get_frequency_distribution(area_original2, array_lmdw[:,2]))
        rrdw1_mse2.append( get_frequency_distribution(area_original2, array_rrdw1[:,2]))
        rrdw2_mse2.append( get_frequency_distribution(area_original2, array_rrdw2[:,2]))
        urfmgrrdw_mse2.append(get_frequency_distribution(area_original2,  array_urfmgrrdw[:,2]))

    # display
    # attribute 1
    fig= plt.figure(figsize=(10, 6),dpi=100)  # 创建一个图形对象和一个子图对象
    bax = brokenaxes(ylims=((0, 100), (300, 700)), hspace=.05, despine=False)



    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    xAxisfont = {'family': 'Times New Roman', 'size': 20,'style': 'italic'}  # 你可以调整size的值来改变字体大小
    xyAxisFont = FontProperties(fname=r'./times.ttf', size=20)  # 步骤2
    legendFont = FontProperties(fname=r'./times.ttf', size=12)

    bax.plot(epsilons, lmdw_mse1,'x',lw=1,color='green',linestyle='-',label='LMDW')
    bax.plot(epsilons, rrdw1_mse1,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    bax.plot(epsilons, rrdw2_mse1,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    bax.plot(epsilons, urfmgrrdw_mse1,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')
    plt.yscale('log')

    bax.set_xlabel(r'$\epsilon$',fontproperties=xyAxisFont)
    bax.set_ylabel(r'$log$ (MSE) of frequency estimation',fontproperties=xyAxisFont)
    if delta==0:
        bax.set_title(r'Attribute 1 of %s ($\delta=%d$, $p_2=%.1f$, ratio=%.2f)' % (datatype, delta,p2,sen_ratio), fontproperties=xyAxisFont)
    else:
        bax.set_title(r'Attribute 1 of %s ($\delta=10^{%d}$,$p_2=%.1f$, ratio=%.2f)' % (datatype,power,p2,sen_ratio),fontproperties=xyAxisFont)
    bax.legend(prop=legendfont)
    bax.grid()#生成网格

    # plt.xlim(0.2,1)
    # #plt.xlim(1, 10)
    # x1_label = axes.get_xticklabels()
    # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    # y1_label = axes.get_yticklabels()
    # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    #
    #
    # axes.tick_params(axis='y',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    # axes.tick_params(axis='x',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    if SaveFileType == 'pdf':
        filename='figures/%s/att1_frequency_logmse_epsilon0_1_%s.pdf' % (datatype,datatype)
    elif SaveFileType=='svg':
        filename = 'figures/%s/att1_frequency_logmse_epsilon0_1_%s.svg' % (datatype,datatype)
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()

    # attribute 2
    fig= plt.figure(figsize=(10, 6),dpi=100)  # 创建一个图形对象和一个子图对象
    bax = brokenaxes(ylims=((200, 300), (400, 700)), hspace=.05, despine=False)

    bax.plot(epsilons, lmdw_mse2,'x',lw=1,color='green',linestyle='-',label='LMDW')
    bax.plot(epsilons, rrdw1_mse2,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    bax.plot(epsilons, rrdw2_mse2,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    bax.plot(epsilons, urfmgrrdw_mse2,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')
    plt.yscale('log')

    bax.set_xlabel(r'$\epsilon$',fontproperties=xyAxisFont)
    bax.set_ylabel(r'$log$ (MSE) of frequency estimation',fontproperties=xyAxisFont)
    if delta==0:
        bax.set_title(r'Attribute 2 of %s ($\delta=%d$, $p_2=%.1f$, ratio=%.2f)' % (datatype, delta,p2,sen_ratio), fontproperties=xyAxisFont)
    else:
        bax.set_title(r'Attribute 2 of %s ($\delta=10^{%d}$, $p_2=%.1f$, ratio=%.2f)' % (datatype,power,p2,sen_ratio),fontproperties=xyAxisFont)
    bax.legend(prop=legendfont)
    bax.grid()#生成网格

    # plt.xlim(0.2,1)
    # #plt.xlim(1, 10)
    # x1_label = axes.get_xticklabels()
    # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    # y1_label = axes.get_yticklabels()
    # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    #
    #
    # axes.tick_params(axis='y',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    # axes.tick_params(axis='x',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    if SaveFileType == 'pdf':
        filename='figures/att2_frequency_logmse_epsilon0_1_%s.pdf' % datatype
    elif SaveFileType == 'svg':
        filename = 'figures/att2_frequency_logmse_epsilon0_1_%s.svg' % datatype
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()

#istribution_estimation_logmse_diff_epsilon(df,watermark,Ks)


def distribution_estimation_mse_diff_p2(df,watermark,Ks):
    array_data=np.array(df)
    max_age = np.max(array_data[:,1])
    min_age = np.min(array_data[:,1])
    min_workinghour = np.min(array_data[:,2])
    max_workinghour = np.max(array_data[:,2])
    age_points = np.linspace(min_age,max_age,10)
    workinghour_points = np.linspace(min_workinghour,max_workinghour,10)
    age_points = np.round(age_points).astype(int)
    workinghour_points = np.round(workinghour_points).astype(int)

    area_original1 = array_data[:, 1]
    area_original2 = array_data[:, 2]

    #epsilons = [0.2, 0.4, 0.6, 0.8, 1.0]
    p2_list = [i * 0.1 for i in range(1, 10)]
    #epsilons = [1*i for i in range(1,11)]
    lmdw_mse1=[]
    rrdw1_mse1=[]
    rrdw2_mse1=[]
    urrdw_mse1=[]
    urfmgrrdw_mse1=[]

    lmdw_mse2=[]
    rrdw1_mse2=[]
    rrdw2_mse2=[]
    urrdw_mse2=[]
    urfmgrrdw_mse2=[]

    for p2 in p2_list:

        lmdw_perturbed_database, mark = embeddedLMDW(epsilon, df,watermark,Ks)
        rrdw1_perturbed_database, mark = embeddedRRDW1(epsilon,df, watermark,Ks)
        rrdw2_perturbed_database, mark = embeddedRRDW2(epsilon,df, watermark,Ks)
        #urrdw_perturbed_database, mark = URRDW(epsilon, df, sen_ratio,watermark,Ks)
        urfmgrrdw_perturbed_database, mark = URFMGRR(epsilon, delta, df, sen_ratio,p2,watermark, Ks)

        array_lmdw =np.array(lmdw_perturbed_database)
        area_lmdw = array_lmdw[:,1]

        array_rrdw1 = np.array(rrdw1_perturbed_database)
        area_rrdw1 = array_rrdw1[:,1]

        array_rrdw2 = np.array(rrdw2_perturbed_database)
        area_rrdw2 = array_rrdw2[:,1]

        #array_urrdw = np.array(urrdw_perturbed_database)
        #area_urrdw = array_urrdw[:,1]

        array_urfmgrrdw = np.array(urfmgrrdw_perturbed_database)
        area_urfmgrrdw = array_urfmgrrdw[:,1]

        lmdw_mse1.append(get_frequency_distribution(area_original1, area_lmdw))
        rrdw1_mse1.append(get_frequency_distribution(area_original1, area_rrdw1))
        rrdw2_mse1.append(get_frequency_distribution(area_original1, area_rrdw2))
        urfmgrrdw_mse1.append(get_frequency_distribution(area_original1, area_urfmgrrdw))

        lmdw_mse2.append(get_frequency_distribution(area_original2, array_lmdw[:,2]))
        rrdw1_mse2.append( get_frequency_distribution(area_original2, array_rrdw1[:,2]))
        rrdw2_mse2.append( get_frequency_distribution(area_original2, array_rrdw2[:,2]))
        urfmgrrdw_mse2.append(get_frequency_distribution(area_original2,  array_urfmgrrdw[:,2]))

    # display
    #----------------------------------------attribute 1---------------------------------------------------------#
    fig= plt.figure(figsize=(10, 6),dpi=100)  # 创建一个图形对象和一个子图对象
    bax = brokenaxes(ylims=((0, 50), (300, 375)), hspace=.05, despine=False)

    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    xAxisfont = {'family': 'Times New Roman', 'size': 20,'style': 'italic'}  # 你可以调整size的值来改变字体大小
    xyAxisFont = FontProperties(fname=r'./times.ttf', size=20)  # 步骤2
    legendFont = FontProperties(fname=r'./times.ttf', size=12)

    bax.plot(p2_list, lmdw_mse1,'x',lw=1,color='green',linestyle='-',label='LMDW')
    bax.plot(p2_list, rrdw1_mse1,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    bax.plot(p2_list, rrdw2_mse1,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    bax.plot(p2_list, urfmgrrdw_mse1,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')


    bax.set_xlabel(r'$p$',fontproperties=xyAxisFont)
    bax.set_ylabel(r'$MSE$ of frequency estimation',fontproperties=xyAxisFont)
    if delta==0:
        bax.set_title(r'Attribute 1 of %s ($\epsilon=%.1f$, $\delta=%d$, ratio=%.1f)' % (datatype, epsilon,delta,sen_ratio), fontproperties=xyAxisFont)
    else:
        bax.set_title(r'Attribute 1 of %s ($\epsilon=%.1f$, $\delta=10^{%d}$, ratio=%.1f)' % (datatype,epsilon,power,sen_ratio),fontproperties=xyAxisFont)
    bax.legend(prop=legendfont)
    bax.grid()#生成网格

    #plt.xlim(0.1,0.9)
    #plt.xlim(1, 10)
    # x1_label = axes.get_xticklabels()
    # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    # y1_label = axes.get_yticklabels()
    # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]


    # bax.tick_params(axis='y',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    # bax.tick_params(axis='x',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    if SaveFileType == 'pdf':
        filename='figures/att1_frequency_mse_p2_0_1_%s.pdf' % datatype
    elif SaveFileType=='svg':
        filename = 'figures/att1_frequency_mse_p2_0_1_%s.svg' % datatype
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()

    #----------------------------------------attribute 2---------------------------------------------------------#
    fig = plt.figure(figsize=(10, 6),dpi=100)  # 创建一个图形对象和一个子图对象
    bax = brokenaxes(ylims=((200, 275), (425, 475)), hspace=.05, despine=False)

    bax.plot(p2_list, lmdw_mse2,'x',lw=1,color='green',linestyle='-',label='LMDW')
    bax.plot(p2_list, rrdw1_mse2,'D',lw=1,color='m',linestyle='-', label='RRDW1')
    bax.plot(p2_list, rrdw2_mse2,'o',lw=1,color='blue',linestyle='-',label='RRDW2')
    #axes.plot(epsilons, urrdw,'s',lw=2,color='green',linestyle='-',label='URRDW')
    bax.plot(p2_list, urfmgrrdw_mse2,'s',lw=1,color='red',linestyle='-', label='URFMGRRDW')


    bax.set_xlabel(r'$p$',fontproperties=xyAxisFont)
    bax.set_ylabel(r'$MSE$ of frequency estimation',fontproperties=xyAxisFont)
    if delta==0:
        bax.set_title(r'Attribute 2 of %s ($\epsilon=%.1f$, $\delta=%d$,  ratio=%.1f)' % (datatype, epsilon,delta,sen_ratio), fontproperties=xyAxisFont)
    else:
        bax.set_title(r'Attribute 2 of %s ($\epsilon=%.1f$, $\delta=10^{%d}$, ratio=%.1f)' % (datatype,epsilon,power,sen_ratio),fontproperties=xyAxisFont)
    bax.legend(prop=legendfont)
    bax.grid()#生成网格

    # plt.xlim(0.1,0.9)
    # #plt.xlim(1, 10)
    # x1_label = axes.get_xticklabels()
    # [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    # y1_label = axes.get_yticklabels()
    # [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    #
    #
    # axes.tick_params(axis='y',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    # axes.tick_params(axis='x',
    #                  labelsize=16,  # y轴字体大小设置
    #                  color='black',  # y轴标签颜色设置
    #                  labelcolor='black',  # y轴字体颜色设置
    #                  direction='in'  # y轴标签方向设置
    #                  )
    if SaveFileType == 'pdf':
        filename='figures/att2_frequency_mse_p2_0_1_%s.pdf' % datatype
    elif SaveFileType == 'svg':
        filename = 'figures/att2_frequency_mse_p2_0_1_%s.svg' % datatype
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()

#distribution_estimation_mse_diff_p2(df,watermark,Ks)


#--------------------------------analyse watermark robustness--------------------------------#
def compare_ber(epsilon, df, watermark, Ks):
    ratio_list=[1*i for i in range(0,100,10) ]
    #ratio_list=[90]
    lmdw_ber=[[],[]]
    rrdw1_ber=[[],[]]
    rrdw2_ber=[[],[]]
    urrdw_ber=[[],[]]
    urfmgrrdw_ber=[[],[]]
    # lmdw_perturb_database, lmdw_mark = embeddedLMDW(epsilon, df, watermark, Ks)
    # rrdw1_perturb_database, rrdw1_mark = embeddedRRDW1(epsilon, df, watermark, Ks)
    # rrdw2_perturb_database, rrdw2_mark = embeddedRRDW2(epsilon, df, watermark, Ks)
    # urrdw_perturb_database, urrdw_mark = URRDW(epsilon, df, watermark, Ks)
    # urfmgrrdw_perturb_database, urfmgrrdw_mark = URFMGRR(epsilon, 0.2, df, watermark, Ks)
    # perturbed_databases=[lmdw_perturb_database,rrdw1_perturb_database,rrdw2_perturb_database,urrdw_perturb_database,urfmgrrdw_perturb_database]
    # #perturbed_databases = [lmdw_perturb_database]
    for ratio in ratio_list:

        lmdw_perturb_database, lmdw_mark = embeddedLMDW(epsilon, df, watermark, Ks)
        rrdw1_perturb_database, rrdw1_mark = embeddedRRDW1(epsilon, df, watermark, Ks)
        rrdw2_perturb_database, rrdw2_mark = embeddedRRDW2(epsilon, df, watermark, Ks)
        urrdw_perturb_database, urrdw_mark = URRDW(epsilon, df,sen_ratio, watermark, Ks)
        urfmgrrdw_perturb_database, urfmgrrdw_mark = URFMGRR(epsilon, delta, df,sen_ratio,p2, watermark, Ks)
        perturbed_databases=[lmdw_perturb_database,rrdw1_perturb_database,rrdw2_perturb_database,urrdw_perturb_database,urfmgrrdw_perturb_database]
        #perturbed_databases=[lmdw_perturb_database,rrdw1_perturb_database,rrdw2_perturb_database,urrdw_perturb_database]
        #perturbed_databases = [lmdw_perturb_database]

        for id in range(len(perturbed_databases)):
            deleted_database = delete_attack(perturbed_databases[id], ratio)
            modified_database = modify_attack(perturbed_databases[id],ratio)
            if id ==0:
                extracted_deleted_watermark = LMDWextraction(df,deleted_database, lmdw_mark, Ks)
                extracted_modified_watermark = LMDWextraction(df,modified_database, lmdw_mark, Ks)

                ber_deleted = bit_error_rate(watermark, extracted_deleted_watermark)
                ber_modified = bit_error_rate(watermark, extracted_modified_watermark)
                lmdw_ber[0].append(ber_deleted)
                lmdw_ber[1].append(ber_modified)
            elif id ==1:
                extracted_deleted_watermark = RRDW1extraction(df,deleted_database, rrdw1_mark, Ks)
                extracted_modified_watermark = RRDW1extraction(df,modified_database, rrdw1_mark, Ks)
                ber_deleted=bit_error_rate(watermark,extracted_deleted_watermark)
                ber_modified=bit_error_rate(watermark,extracted_modified_watermark)
                rrdw1_ber[0].append(ber_deleted)
                rrdw1_ber[1].append(ber_modified)
            elif id ==2:
                extracted_deleted_watermark = RRDW2extraction(df,deleted_database, rrdw2_mark, Ks)
                extracted_modified_watermark = RRDW2extraction(df,modified_database, rrdw2_mark, Ks)
                ber_deleted=bit_error_rate(watermark,extracted_deleted_watermark)
                ber_modified=bit_error_rate(watermark,extracted_modified_watermark)
                rrdw2_ber[0].append(ber_deleted)
                rrdw2_ber[1].append(ber_modified)
            elif id ==3:
                extracted_deleted_watermark = URRDWextraction(df,deleted_database, urrdw_mark, Ks)
                extracted_modified_watermark = URRDWextraction(df,modified_database, urrdw_mark, Ks)
                ber_deleted=bit_error_rate(watermark,extracted_deleted_watermark)
                ber_modified=bit_error_rate(watermark,extracted_modified_watermark)
                urrdw_ber[0].append(ber_deleted)
                urrdw_ber[1].append(ber_modified)
            elif id ==4:
                extracted_deleted_watermark = URFMGRR_extraction(df,deleted_database, urfmgrrdw_mark, Ks)
                extracted_modified_watermark = URFMGRR_extraction(df,modified_database, urfmgrrdw_mark, Ks)

                ber_deleted=bit_error_rate(watermark,extracted_deleted_watermark)
                ber_modified=bit_error_rate(watermark,extracted_modified_watermark)
                urfmgrrdw_ber[0].append(ber_deleted)
                urfmgrrdw_ber[1].append(ber_modified)

    fig, axes = plt.subplots(figsize=(10, 6), dpi=100)  # 创建一个图形对象和一个子图对象

    legendfont = {'family': 'Times New Roman', 'size': 12}  # 你可以调整size的值来改变字体大小
    xyAxisFont = FontProperties(fname=r'./times.ttf', size=20)  # 步骤2
    legendFont = FontProperties(fname=r'./times.ttf', size=12)

    axes.plot(ratio_list, lmdw_ber[0],'x',lw=2,color='green',linestyle='-',label='LMDW (deletion)')
    axes.plot(ratio_list, lmdw_ber[1],'D',lw=2,color='green',linestyle='-',label='LMDW (alteration)')

    axes.plot(ratio_list, rrdw1_ber[0],'x',lw=2,color='m',linestyle='-', label='RRDW1  (deletion)')
    axes.plot(ratio_list, rrdw1_ber[1], 'D', lw=2, color='m', linestyle='-', label='RRDW1 (alteration)')

    axes.plot(ratio_list, rrdw2_ber[0],'x',lw=2,color='blue',linestyle='-',label='RRDW2 (deletion)')
    axes.plot(ratio_list, rrdw2_ber[0], 'D', lw=2, color='blue', linestyle='-', label='RRDW2 (alteration)')

    #axes.plot(ratio_list, urrdw_ber[0],'x',lw=2,color='green',linestyle='-',label='URRDW(deletion)')
    #axes.plot(ratio_list, urrdw_ber[1], 'D', lw=2, color='green', linestyle='-', label='URRDW(alteration)')

    axes.plot(ratio_list, urfmgrrdw_ber[0],'x',lw=2,color='red',linestyle='-', label='URFMGRRDW (deletion)')
    axes.plot(ratio_list, urfmgrrdw_ber[1], 'D', lw=2, color='red', linestyle='-', label='URFMGRRDW (alteration)')

    axes.set_xlabel('Deleted/Altered Tuples',fontproperties=xyAxisFont)
    axes.set_ylabel('BER',fontproperties=xyAxisFont)
    if delta==0:
        axes.set_title(r'%s ($\epsilon=%d,\delta= %d$, $p_2=%.1f$, ratio=%.2f)' % (datatype, epsilon, delta,p2,sen_ratio), fontproperties=xyAxisFont)
    else:
        axes.set_title(r'%s ($\epsilon=%d,\delta= 10^{%d}$, $p_2=%.1f$, ratio=%.2f)' % (datatype,epsilon,power,p2,sen_ratio),fontproperties=xyAxisFont)
    axes.legend(prop=legendfont)
    axes.grid()#生成网格

    plt.xlim(0,90)
    x1_label = axes.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = axes.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]


    axes.tick_params(axis='y',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    axes.tick_params(axis='x',
                     labelsize=16,  # y轴字体大小设置
                     color='black',  # y轴标签颜色设置
                     labelcolor='black',  # y轴字体颜色设置
                     direction='in'  # y轴标签方向设置
                     )
    if SaveFileType=='pdf':
       filename='figures/ber_%s.pdf' % datatype
    elif SaveFileType=='svg':
       filename = 'figures/ber_%s.svg' % datatype
    plt.savefig(filename,dpi=400,bbox_inches = 'tight')
    plt.show()

#compare_ber(epsilon, df, watermark, Ks)


# for select in range(6):
#
#     df,array_data,datatype=select_dataset(select)
#     figureshow_diff_epsilon(df,watermark,Ks)
#     figureshow_diff_delta(df,watermark,Ks)
#     distribution_estimation_mse(df, watermark, Ks)
#     compare_ber(epsilon, df, watermark, Ks)