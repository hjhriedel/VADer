#%%
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from utils.utils import process_config
import models.flexNet_00 as fn
import math
import scipy.signal as sig
import pandas as pd
import json
import seaborn as sns   
from tqdm import tqdm
import pickle
from glob import glob
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

def plot_results(pred, peaks, tv, plot):
    data = np.load('1961.npz', allow_pickle=True)
    raw = data['x'][:,plot[1]]
    raw /= raw.max()
    pred, tv = pred.flatten()[:len(raw)], tv[:len(raw)]
    t = np.arange(len(pred))
    fig, axs = plt.subplots(1, figsize=(3, 4))
    axs.plot(t/600, raw,zorder=2, lw=0.5, color='navy', label='raw signal')
    # plt.xlabel(f'Time [s]')
    plt.xlim([0.1, 0.6])
    # plt.yticks([1,0,-1])
    # plt.xticks([0,0.5,1])
    # axs.set_ylabel(f'Acceleration [a.u.]')
    
    plt.tick_params(labelleft = False ,labelbottom = False)
    plt.tight_layout()
    plt.savefig(f"sig{plot[2]}_{plot[1]}.png",dpi=1000)
    plt.show(block=False)
    
    fig, axs = plt.subplots(1, figsize=(3, 4))
    l2, = axs.plot(t/600, pred,zorder=2, lw=0.5, label='prediction')
    # l3 = axs.plot([t[tv>0]/600,t[tv>0]/600], [0,1], color='r',zorder=0, lw=0.5, label='ground truth')
    # l4 = axs.plot([peaks/600], [pred[peaks]], 'v', color='m',zorder=0, lw=0.5, label='detected axle')
    # plt.xlabel(f'Time [s]')
    plt.xlim([0.1, 0.6])
    # axs.set_ylabel(f'Prediction')    
    # plt.xticks([0,0.5,1])
    # fig.legend(handles = [l2, l3[0], l4[0]])
    plt.tick_params(labelleft = False ,labelbottom = False)
    plt.tight_layout()
    plt.savefig(f"pred{plot[2]}_{plot[1]}.png",dpi=1000)
    plt.show(block=False)
    
    fig, axs = plt.subplots(1, figsize=(3, 4))
    l2, = axs.plot(t/600, pred,zorder=2, lw=0.5, label='prediction')
    # l3 = axs.plot([t[tv>0]/600,t[tv>0]/600], [0,1], color='r',zorder=0, lw=0.5, label='ground truth')
    l4 = axs.plot([peaks/600], [pred[peaks]], 'v', color='m',zorder=0, markersize=15, label='detected axle')
    # plt.xlabel(f'Time [s]')
    plt.xlim([0.1, 0.6])
    # axs.set_ylabel(f'Prediction')    
    # plt.xticks([0,0.5,1])
    # fig.legend(handles = [l2, l3[0], l4[0]])
    plt.tick_params(labelleft = False ,labelbottom = False)
    plt.tight_layout()
    plt.savefig(f"peak{plot[2]}_{plot[1]}.png",dpi=1000)
    plt.show(block=False)
    plt.clf()
    

def plot_json():
    json_file = "C:/Users/Henrik Riedel/Desktop/val_recall_vs_val_precision_chart_data.json"
    with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)
    df = pd.DataFrame(config_dict)
    df = df.sort_values('gamma').iloc[2:-4]
    ax = sns.scatterplot(x="precision", y="recall", data=df, 
                    palette="rocket_r",s= 80, hue="gamma", ci=None, legend=False)
    
    plt.xlim(0.882,0.98)
    plt.savefig('gamma.png',dpi=600, bbox_inches="tight")

def error_axle_positions(prediction, true_values, v, threshold=20, distance=1.0, plot=[False], thresholdInSamples=True):

    # defining number and position of true axles:
    idx_axle_true = np.where(true_values == 1)[0]
    num_axle_true = np.count_nonzero(true_values)
    distanceInSample = distance if thresholdInSamples else math.floor(distance / np.mean(v) * 600)

    # defining number and position of predicted axles:
    idx_axle_pred, _ = sig.find_peaks(prediction, height=0.25, prominence=0.15, distance=distanceInSample)
    num_axle_pred = idx_axle_pred.shape[0]
    if plot[0]:
        plot_results(prediction, idx_axle_pred, true_values, plot)

    num_axle_correct, num_axle_false = 0, 0
    false_samples, false_m = np.zeros_like(idx_axle_true), np.zeros_like(idx_axle_true, dtype=np.float64)
    # axle_correct = np.zeros_like(true_values)
    acc_per_axle = []
    for i, _v in zip(range(num_axle_true),v):
        if idx_axle_pred.any():
            idx_min_distance = np.argmin(abs(idx_axle_pred - idx_axle_true[i]))
            thresholdInSample = threshold if thresholdInSamples else threshold / _v * 600
            if min(abs(idx_axle_pred - idx_axle_true[i])) <= thresholdInSample:
                false_samples[i-num_axle_false] = idx_axle_true[i] - idx_axle_pred[idx_min_distance]
                false_m[i-num_axle_false] = (idx_axle_true[i] - idx_axle_pred[idx_min_distance]) / 600 * _v
                num_axle_correct += 1
                idx_axle_pred = np.delete(idx_axle_pred, idx_min_distance)
                # axle_correct[idx_axle_true[i]] = 1
                acc_per_axle.append([i,True])
            else:
                false_samples = np.delete(false_samples, i-num_axle_false, axis=0)
                false_m = np.delete(false_m, i-num_axle_false, axis=0)
                num_axle_false += 1
                acc_per_axle.append([i,False])
        else:
            false_samples = np.delete(false_samples, i - num_axle_false, axis=0)
            false_m = np.delete(false_m, i - num_axle_false, axis=0)
            num_axle_false += 1
            acc_per_axle.append([i,False])
    return num_axle_true, num_axle_pred, num_axle_correct, false_samples, false_m, np.array(acc_per_axle)

def error_statistics_2(num_axle_true, num_axle_pred, num_axle_correct):    
    recall = num_axle_correct / num_axle_true
    precision = num_axle_correct / num_axle_pred if num_axle_pred > 0 else 0
    if recall == 0 and precision == 0:
        return 0, 0, 0
    else:
        return 2*precision*recall/(precision+recall), precision, recall

def violinplot(precision, recall):
    df = pd.DataFrame()
    length = len(np.hstack(precision))
    df['value']=np.hstack((np.hstack(precision),np.hstack(recall)))
    df['metric']=np.hstack((['precision',]*length,['recall',]*length))
    thresholds = np.hstack(np.array([[200,37,20],]*(len(recall[0]))).T)
    df['threshold']=np.hstack((thresholds,thresholds))
    fig, ax = plt.subplots(1, 1, figsize=[8,5])
    ax = sns.violinplot(x="threshold", y='value', hue='metric',
                        data=df, inner="quartile", split=True, cut=0, scale="count")
    plt.xlabel(f'Threshold [cm]')
    plt.ylabel(f'Score')
    ax.set_ylim(0,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
    ax.spines['bottom'].set_visible(False)     
    plt.tick_params(bottom = False) 
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('pr.png',dpi=600, bbox_inches="tight")
    
    
def plot_norm_disp(dir, false_samples, name="", unit='samples'):
    mean = np.mean(false_samples)
    std = np.std(false_samples)
    h = sorted(np.arange(mean-std*4,mean+std*4,0.5))
    normal = stats.norm.pdf(h, mean, std)
    plt.figure(1)
    # plt.plot(h,normal) 
    plt.hist(false_samples, bins=39*2+1, density=True)
    plt.xlabel(f'Spatial Error [{unit}]')
    plt.xticks([-200,-100,-50,-25,0,25,50,100,200],fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel(f'Density')
    plt.savefig(f'{name}{dir}_norm.png', dpi=300)
    # plt.show(block=False)
    plt.clf()

def plotPerAxle(acc_per_axle, title):
    df2 = pd.DataFrame(data = acc_per_axle, columns=['Axle Number','Accuracy'])
    fig, ax1 = plt.subplots()
    ax1 = sns.lineplot(x="Axle Number", y="Accuracy", data=df2, marker = 'o', ax=ax1)
    ax2 = ax1.twinx()
    plot_df = df2['Axle Number'].value_counts()
    plot_df = plot_df.sort_index()
    ax2.plot(plot_df.index, plot_df.values, color='orange')
    ax2.set_ylabel('Occurrence of Axle Number')
    plt.title(title)
    plt.show(block=False)
    
# data = np.load("test.npz", allow_pickle=True)
# x = data['x']
# y = data['y']
# data.close
# print(x[0].shape)

dPATH = "D:/Henrik Riedel/Data TrigG1 new Code 4/"
def test_evaluation(results, thresholds=[2.0], inSamples=True, PLOT=False, NAME='', without_axle=10):
    acc_per_axle = np.ndarray([0,2])
    data = []
    for threshold in thresholds:
        i = 0
        for result in results:
            if result[1]==without_axle:
                continue
            # acc_list, mean_fs_list, std_fs_list, avg_neg_list, avg_pos_list, percent_null_list, percent_neg_list, percent_pos_list, recall_list = [], [], [], [], [], [], [], [], []
            threshold = 20 if inSamples else threshold # false_m in meter from exact position
            distance = 20 if inSamples else 2.0 # distance in meter of axles inbetween a bogie
            _num_axle_true, _num_axle_pred, _num_axle_correct, _fs, _fs_m, _acc_per_axle = error_axle_positions(result[2], result[3], np.abs(result[4]), 
                                    threshold=threshold, distance=distance, plot=[PLOT,i,NAME], thresholdInSamples=inSamples)
            acc_per_axle = np.vstack((acc_per_axle,_acc_per_axle))
            
            _f1, _precision, _recall = error_statistics_2(_num_axle_true, _num_axle_pred, _num_axle_correct)
            
            data.append([threshold, result[0][:-4], result[1], int(result[3].sum()), _num_axle_correct, _num_axle_pred, result[4].mean(),
                         _fs, _fs_m, _f1, _precision, _recall, abs(_fs_m).mean()*100, abs(_fs).mean(), [a[0] for a in _acc_per_axle], [a[1] for a in _acc_per_axle]])
            
            i+=1
    return data, acc_per_axle
             
#%%
from tqdm import tqdm
from joblib import Parallel, delayed
tot_data, name = [], []

def get_results(f):
    print(f[19:-8])
    with open(f, "rb") as data:
        results = pickle.load(data)
    data_new, _ = test_evaluation(results, inSamples=False, thresholds=[.37,2], without_axle=11)
    return data_new, f[19:-8]


# if __name__ == '__main__':
# results_tot = Parallel(n_jobs=7)(delayed(get_results)(f) for f in glob("VADER results/*.bin"))

# import pickle
# with open('VADERresults.bin', "wb") as f:
#     pickle.dump(results_tot, f)

import pickle
with open('VADERresults.bin', "rb") as f:
    results_tot = pickle.load(f)

#%%

temp = []

for _res in results_tot:
    df = pd.DataFrame(data = _res[0], columns=['Grenzwert in m', 'Überfahrt', 'Sensor', 'n Achsen', 'korrekte Achsen', 'vorhergesagte Achsen', 'Geschwindigkeiten', 'Abweichung Samples', 'Abweichung Meter', 'F1', 'Precision', 'Recall', 'Durschnitt cm', 'Durschnitt Sample', 'AxId', 'AccPerAxle'])
    df["Run"] = np.repeat(_res[1],len(df))
    temp.append(df)

df = pd.concat(temp)

#%%

df["fold"] = df.Run.str.split(" ").str[-1]
df.fold = df.fold.astype(int)
df["Model"] = df.Run.str.split(" ").str[0]
df["Type"] = None
df.Type[df.fold > 99] = "stratified"
df.Type[df.fold < 99] = "DGPS"


print(f"Züge: {len(df[df.Run=='VADER 100'])//10} | Achsen: {df[df.Run=='VADER 100']['n Achsen'].sum()//10}")
print(f"Züge: {len(df[df.Run=='VADER 1'])//10} | Achsen: {df[df.Run=='VADER 1']['n Achsen'].sum()//10}")
#%%

def get_stats(df, sensor=11):
    sort = ["Grenzwert in m", "Type", "Model", "fold"]
    _df = df[df["Sensor"] != sensor].copy()
    keys = _df.groupby(sort).groups.keys()

    tp = _df.groupby(sort)['korrekte Achsen'].sum().to_numpy()
    fp = (_df.groupby(sort)['vorhergesagte Achsen'].sum()-tp).to_numpy()
    fn = (_df.groupby(sort)['n Achsen'].sum()-tp).to_numpy()
    f1 = tp/(tp+fp/2+fn/2)

    data = []
    for _f1, _type in zip(f1,keys):
        temp = [_f1]
        temp.extend(_type)
        data.append(temp)

    df_results = pd.DataFrame(data, columns=["F1", "Grenzwert in m", "Type", "Model", "fold"])

    sort2 = ["Grenzwert in m", "Type", "Model"]
    filt = ["Grenzwert in m", "Type", "Model","Abweichung Meter"]
    df2 = _df[filt].copy().explode("Abweichung Meter")
    SE = df2.groupby(sort2)["Abweichung Meter"].apply(lambda c: c.abs().mean())
    SESTD = df2.groupby(sort2)["Abweichung Meter"].std().to_numpy()
    keys = _df.groupby(sort2).groups.keys()
        
    data = []
    for _type, _se, _sestd in zip(keys,SE,SESTD):
        temp = [_se, _sestd]
        temp.extend(_type)
        data.append(temp)

    _df_results = pd.DataFrame(data, columns=["SE", "SESTD", "Grenzwert in m", "Type", "Model"])
        
    sort = ["Grenzwert in m", "Type", "Model"]
    df_results_tot = df_results.merge(_df_results, how='left', on=["Grenzwert in m", "Type", "Model"])
    # print(df_results_tot.groupby(sort).agg({'F1':["mean", "std"],'SE':"mean",'SESTD':'mean'}))
    return df_results_tot
#%%    
# sort = ["Threshold in cm", "Type", "Model"]
# results = get_stats(df[df.Type == "DGPS"], axle=11)
# results['Threshold in cm'] = results['Grenzwert in m'] * 100
# results['F1 score in %'] = results['F1']
# results['SE in cm'] = results['SE']
# results['STD of SE in cm'] = results['SESTD']
# print((results.groupby(sort).agg({'F1 score in %':["mean", "std"],
#                                   'SE in cm':"mean",
#                                   'STD of SE in cm':'mean'})*100))#.to_latex(float_format="%.3g"))
#%%

sort = ["Threshold in cm", "Type", "Model"]
results = get_stats(df)
results['Threshold in cm'] = results['Grenzwert in m'] * 100
results = results.groupby(sort).agg({'F1':["mean", "std"],
                                  'SE':"mean",
                                  'SESTD':'mean'}).reset_index()

_results = results.melt(id_vars=["Threshold in cm", "Type", "Model"])
_results['metric'] = _results['variable_0'] + " " + _results['variable_1']
_results['metric'] = _results['metric'].replace({'SE mean':'Spatial Error mean',
                       'SESTD mean':'Spatial Error STD',})
_results = pd.pivot_table(_results, values=["value"], index=["Threshold in cm", "Type",'metric'], columns=['Model'])*100
_results['Improvement'] = 0
_results.iloc[::4,2] = (1 - (100-_results.iloc[::4,1]) / (100-_results.iloc[::4,0]) )*100
_results.iloc[1::4,2] = (1 - (_results.iloc[1::4,1]) / (_results.iloc[1::4,0]) ) * 100
_results.iloc[2::4,2] = (1 - (_results.iloc[2::4,1]) / (_results.iloc[2::4,0]) ) * 100
_results.iloc[3::4,2] = (1 - (_results.iloc[3::4,1]) / (_results.iloc[3::4,0]) ) * 100
print(_results.to_latex(float_format="%.3g"))
for i in range(4):
    print(_results.iloc[i::4,2].mean())
    
#%%
# results7 = get_stats(df, axle=7)
# results7['Threshold in cm'] = results7['Grenzwert in m'] * 100
# results7['F1 score in %'] = results7['F1']
# results7['SE in cm'] = results7['SE']
# results7['STD of SE in cm'] = results7['SESTD']
# print((results7.groupby(sort).agg({'F1 score in %':["mean", "std"],
#                                   'SE in cm':"mean",
#                                   'STD of SE in cm':'mean'})*100))#.to_latex(float_format="%.3g"))
# %%


results = get_stats(df, sensor=7)
results['Threshold in cm'] = results['Grenzwert in m'] * 100
results = results.groupby(sort).agg({'F1':["mean", "std"],
                                  'SE':"mean",
                                  'SESTD':'mean'}).reset_index()
_results = results.melt(id_vars=["Threshold in cm", "Type", "Model"])
_results['metric'] = _results['variable_0'] + " " + _results['variable_1']
_results['metric'] = _results['metric'].replace({'SE mean':'Spatial Error mean',
                       'SESTD mean':'Spatial Error STD',})
                                                 
_results = pd.pivot_table(_results, values=["value"], index=["Threshold in cm", "Type",'metric'], columns=['Model'])*100
_results['Improvement'] = 0
_results.iloc[::4,2] = (1 - (100-_results.iloc[::4,1]) / (100-_results.iloc[::4,0]) )*100
_results.iloc[1::4,2] = (1 - (_results.iloc[1::4,1]) / (_results.iloc[1::4,0]) ) * 100
_results.iloc[2::4,2] = (1 - (_results.iloc[2::4,1]) / (_results.iloc[2::4,0]) ) * 100
_results.iloc[3::4,2] = (1 - (_results.iloc[3::4,1]) / (_results.iloc[3::4,0]) ) * 100
print(_results.to_latex(float_format="%.3g"))
for i in range(4):
    print(_results.iloc[i::4,2].mean())

#%%

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(context='paper', style='ticks', font='serif', palette='colorblind')

plt.figure()#figsize=(15, 10))
ax = sns.boxplot(x="Type", y="F1", data=df[df["Grenzwert in m"]==2], hue="Model", fliersize=0., linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
# plt.ylim(0,1.1)
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
plt.ylabel(r"$F_1$ score in %")
# plt.title("Grenzwert = 2m")
plt.legend(loc="lower left")
plt.savefig("Grenzwert2",dpi=600, bbox_inches="tight")
plt.show(block=False)
#%%
plt.figure()
ax = sns.boxplot(x="Type", y="F1", data=df[df["Grenzwert in m"]==.37], hue="Model", fliersize=0., linewidth=2)
# plt.title("Grenzwert = 37cm")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
# plt.ylim(0,1)
plt.ylabel(r"$F_1$ score in %")
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
plt.legend(loc="lower left")
plt.savefig("Grenzwert37",dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure() # figsize=(12.8, 9.6))
ax = sns.boxplot(x="Sensor", y="F1", data=df[(df["Grenzwert in m"]==2)*(df.Type == "DGPS")], hue="Model", fliersize=0., linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
plt.ylabel(r"$F_1$ score in %")
plt.xlabel("sensor")
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
# plt.ylim(0,1)
plt.xticks([i for i in range(10)],['L1','L2','L3','L4','L5','R1','R2','R3','R4','R5',])
# plt.title("Grenzwert = 2m & single train")
plt.legend(loc="lower left")
plt.savefig("SensorSingleTrain2",dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure() # figsize=(12.8, 9.6))
ax = sns.boxplot(x="Sensor", y="F1", data=df[(df["Grenzwert in m"]==2)*(df.Type == "stratified")], hue="Model", fliersize=0., linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
plt.ylabel(r"$F_1$ score in %")
plt.xlabel("sensor")
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
# plt.ylim(0,1)
plt.xticks([i for i in range(10)],['L1','L2','L3','L4','L5','R1','R2','R3','R4','R5',])
# plt.title("Grenzwert = 2m & all trains")
plt.legend(loc="lower left")
plt.savefig("SensorAllTrains2",dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure() # figsize=(12.8, 9.6))
ax = sns.boxplot(x="Sensor", y="F1", data=df[(df["Grenzwert in m"]==.37)*(df.Type == "DGPS")], hue="Model", fliersize=0., linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
plt.ylabel(r"$F_1$ score in %")
plt.xlabel("sensor")
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
# plt.ylim(0,1)
plt.xticks([i for i in range(10)],['L1','L2','L3','L4','L5','R1','R2','R3','R4','R5',])
# plt.title("Grenzwert = 37cm & single train")
plt.legend(loc="lower left")
plt.savefig("SensorSingleTrain37",dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure() # figsize=(12.8, 9.6))
ax = sns.boxplot(x="Sensor", y="F1", data=df[(df["Grenzwert in m"]==.37)*(df.Type == "stratified")], hue="Model", fliersize=0., linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
plt.ylabel(r"$F_1$ score in %")
plt.xlabel("sensor")
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
# plt.ylim(0,1)
plt.xticks([i for i in range(10)],['L1','L2','L3','L4','L5','R1','R2','R3','R4','R5',])
# plt.title("Grenzwert = 37cm & all trains")
plt.legend(loc="lower left")
plt.savefig("SensorAllTrains37",dpi=600, bbox_inches="tight")
plt.show(block=False)

df["Spatial Accuracy"] = (200 - df["Durschnitt cm"])/200

plt.figure() # figsize=(12.8, 9.6))
ax = sns.boxplot(x="Sensor", y="Spatial Accuracy", data=df[(df["Grenzwert in m"]==2)*(df.Type == "DGPS")], hue="Model", fliersize=0., linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
plt.ylabel("spatial accuracy in %")
plt.xlabel("sensor")
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
# plt.ylim(0,1)
plt.xticks([i for i in range(10)],['L1','L2','L3','L4','L5','R1','R2','R3','R4','R5',])
# plt.title("Grenzwert = 2m & single train")
plt.legend(loc="lower left")
plt.savefig("SensorSingleTrain2SE",dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure() # figsize=(12.8, 9.6))
ax = sns.boxplot(x="Sensor", y="Spatial Accuracy", data=df[(df["Grenzwert in m"]==2)*(df.Type == "stratified")], hue="Model", fliersize=0., linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)   
ax.spines['bottom'].set_visible(False) 
plt.yticks(np.arange(0.,1.01,0.2),[0,20,40,60,80,100])
plt.ylabel("spatial accuracy in %")
plt.xlabel("sensor")
# plt.ylim(0,1)
plt.xticks([i for i in range(10)],['L1','L2','L3','L4','L5','R1','R2','R3','R4','R5',])
# plt.title("Grenzwert = 2m & all trains")
plt.legend(loc="lower left")
plt.savefig("SensorAllTrains2SE",dpi=600, bbox_inches="tight")
plt.show(block=False)


#%%

df = df.explode(["AxId","AccPerAxle"])


_df2 = df[(df["Grenzwert in m"]==2)].copy()

df2 = _df2.groupby(["Model", "fold", "Type", "AxId"]).F1.mean().reset_index()
# plt.figure() # figsize=(12.8, 9.6))
# sns.lineplot(data=df2.reset_index(), x="AxId", y="F1", hue="Model", 
#              style="Type")

#%%

plt.figure() # figsize=(12.8, 9.6))
g = sns.JointGrid()
ax = sns.lineplot(data=df2, x="AxId", y="F1", hue="Model", 
             style="Type", ax=g.ax_joint)
sns.kdeplot(data=df2[df2.Type=="DGPS"], y="F1", hue="Model", fill=True, lw=1, 
            ax=g.ax_marg_y,color=sns.color_palette('colorblind')[0], legend=False)
sns.kdeplot(data=df2[df2.Type=="stratified"], y="F1", hue="Model", fill=True, lw=1, 
            ax=g.ax_marg_y,color=sns.color_palette('colorblind')[0], legend=False)
sns.histplot(data=_df2[(_df2.Type=="stratified")].AxId, fill=False, bins=64, lw=1.5, element="step",
             ax=g.ax_marg_x,color="grey", stat="count", common_norm=False, linestyle="--")
sns.histplot(data=_df2[(_df2.Type=="DGPS")].AxId, fill=False, bins=64, lw=1.5, element="step",
             ax=g.ax_marg_x,color="grey", stat="count", common_norm=False)
sns.move_legend(ax, "lower center", title='', ncol=2, frameon=False)
plt.ylim(0.7,1)
plt.yticks(np.arange(0.7,1.01,0.05),[70,75,80,85,90,95,100])
g.set_axis_labels('axle number', r"$F_1$ score in %")
plt.savefig("AxleNumberVsF1",dpi=600, bbox_inches="tight")
plt.show(block=False)


#%%

df2 = _df2.groupby(["Model", "fold", "Type", "n Achsen"]).F1.mean().reset_index()

plt.figure() # figsize=(12.8, 9.6))
g = sns.JointGrid()
ax = sns.lineplot(data=df2[df2.Type=="DGPS"].reset_index(), x="n Achsen", y="F1", hue="Model", ax=g.ax_joint)
sns.kdeplot(data=df2[df2.Type=="DGPS"].reset_index(), y="F1", hue="Model", fill=True, linewidth=1, cut=0,
            ax=g.ax_marg_y,color=sns.color_palette('colorblind')[0], legend=False)
ax1 = sns.histplot(data=_df2[_df2.Type=="DGPS"]["n Achsen"], fill=False, bins=64, linewidth=1, 
             ax=g.ax_marg_x,color='grey', legend=False)
ax1.set_yticks([])
sns.move_legend(ax, "lower center", title='', ncol=2, frameon=False)
g.set_axis_labels('train length in axles', 'F1')
plt.ylim(0.55, 1)
plt.savefig("LengthAxlesVsF1_single",dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure() # figsize=(12.8, 9.6))
g = sns.JointGrid()
ax = sns.lineplot(data=df2[df2.Type=="stratified"].reset_index(), x="n Achsen", y="F1", hue="Model", ax=g.ax_joint)
sns.kdeplot(data=df2[df2.Type=="stratified"].reset_index(), y="F1", hue="Model", fill=True, linewidth=1, cut=0,
            ax=g.ax_marg_y,color=sns.color_palette('colorblind')[0], legend=False)
ax1 = sns.histplot(data=_df2[_df2.Type=="stratified"]["n Achsen"], fill=False, bins=64, linewidth=1, 
             ax=g.ax_marg_x,color='grey', legend=False)
ax1.set_yticks([])
sns.move_legend(ax, "lower center", title='', ncol=2, frameon=False)
g.set_axis_labels('train length in axles', 'F1')
plt.ylim(0.65, 1)
plt.savefig("LengthAxlesVsF1_all",dpi=600, bbox_inches="tight")
plt.show(block=False)

#%%


import json
# def plot_json():
json_file = "f1_score_chart_data.json"
with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
df = pd.DataFrame(config_dict)

df["Set"] = df.name.str.split("<br>").apply(lambda x: x[0])
df["Architecture"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[0])
df["Run"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[1]).astype(int)
df["Type"] = "DGPS"
df.Type[df.Run > 99] = "stratified"
df.drop(columns=["name", "type"], inplace=True)
df = df.explode(["x", "y"]).reset_index(drop=True)
df = df.replace({"f1":"training", "val_f1":"validation"})
df.head()

#%%

plt.figure()
sns.lineplot(x="x", y="y", data=df[df.Type=="DGPS"], hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.ylim(0.8,1.0)
plt.yticks(np.arange(0.8,1.01,0.05),[80,85,90,95,100])
plt.xlabel("epoch")
plt.ylabel(r"$F_1$ score in %")
plt.xlim(0)
plt.savefig("f1_vad_vader_single.png", dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure()
ax = sns.lineplot(x="x", y="y", data=df[df.Type=="stratified"], hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.ylim(0.8,1.0)
plt.yticks(np.arange(0.8,1.01,0.05),[80,85,90,95,100])
plt.xlabel("epoch")
plt.ylabel(r"$F_1$ score in %")
plt.xlim(0)
plt.savefig("f1_vad_vader_all.png", dpi=600, bbox_inches="tight")
plt.show(block=False)

#%%

import json
# def plot_json():
json_file = "loss_val_loss_test_loss_vs_epoch_chart_data.json"
with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
df = pd.DataFrame(config_dict)

df["Set"] = df.name.str.split("<br>").apply(lambda x: x[0])
df["Architecture"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[0])
df["Run"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[1]).astype(int)
df["Type"] = "DGPS"
df.Type[df.Run > 99] = "stratified"
df.drop(columns=["name", "type"], inplace=True)
df = df.explode(["x", "y"]).reset_index(drop=True)
df = df.replace({"loss":"training", "val_loss":"validation"})
df.head()

#%%

plt.figure()
sns.lineplot(x="x", y="y", data=df[df.Type=="DGPS"], hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.yscale("log")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.xlim(0)
plt.yticks([0.002,0.003, 0.004,0.006,0.008,0.01],[0.002,"",0.004,0.006,0.008,0.01])
plt.savefig("loss_vad_vader_single.png", dpi=600, bbox_inches="tight")
plt.show(block=False)

plt.figure()
sns.lineplot(x="x", y="y", data=df[df.Type=="stratified"], hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.yscale("log")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.yticks([0.002,0.003, 0.004,0.006,0.008,0.01],[0.002,"",0.004,0.006,0.008,0.01])
plt.xlim(0)
plt.savefig("loss_vad_vader_all.png", dpi=600, bbox_inches="tight")
plt.show(block=False)

# %%


json_file = "loss_vad_00_gn.json"
with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
df = pd.DataFrame(config_dict)

df["Set"] = df.name.str.split("<br>").apply(lambda x: x[0])
df["Architecture"] = df.name.str.split("<br>").apply(lambda x: ''.join(x[1].split(" ")[:-1]))
df["Run"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[-1]).astype(int)
df["Type"] = "DGPS"
df.Type[df.Run > 99] = "stratified"
df.drop(columns=["name", "type"], inplace=True)
df = df.explode(["x", "y"]).reset_index(drop=True)
df = df.replace({"loss":"training", "val_loss":"validation"})
df.head()

plt.figure()
sns.lineplot(x="x", y="y", data=df, hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.yscale("log")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.xlim(0)
plt.yticks([0.002,0.003, 0.004,0.006,0.008,0.01],[0.002,"",0.004,0.006,0.008,0.01])
plt.savefig("loss_vad_gn_00.png", dpi=600, bbox_inches="tight")
plt.show(block=False)

json_file = "f1_vad_00_gn.json"
with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
df = pd.DataFrame(config_dict)

df["Set"] = df.name.str.split("<br>").apply(lambda x: x[0])
df["Architecture"] = df.name.str.split("<br>").apply(lambda x: ''.join(x[1].split(" ")[:-1]))
df["Run"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[-1]).astype(int)
df["Type"] = "DGPS"
df.Type[df.Run > 99] = "stratified"
df.drop(columns=["name", "type"], inplace=True)
df = df.explode(["x", "y"]).reset_index(drop=True)
df = df.replace({"f1":"training", "val_f1":"validation"})
df.head()

plt.figure()
sns.lineplot(x="x", y="y", data=df, hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.ylabel(r"$F_1$ score in %")
plt.xlabel("epoch")
plt.yticks(np.arange(0.7,1.01,0.05),[70,75,80,85,90,95,100])
plt.xlim(0)
plt.savefig("f1_vad_gn_00.png", dpi=600, bbox_inches="tight")
plt.show(block=False)
# %%

json_file = "loss_vad_gn.json"
with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
df = pd.DataFrame(config_dict)

df["Set"] = df.name.str.split("<br>").apply(lambda x: x[0])
df["Architecture"] = df.name.str.split("<br>").apply(lambda x: ''.join(x[1].split(" ")[:-1]))
df["Run"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[-1]).astype(int)
df["Type"] = "DGPS"
df.Type[df.Run > 99] = "stratified"
df.drop(columns=["name", "type"], inplace=True)
df = df.explode(["x", "y"]).reset_index(drop=True)
df = df.replace({"loss":"training", "val_loss":"validation"})
df.head()

plt.figure()
sns.lineplot(x="x", y="y", data=df, hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.xlim(0)
plt.savefig("loss_vad_gn.png", dpi=600, bbox_inches="tight")
plt.show(block=False)

json_file = "f1_vad_gn.json"
with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
df = pd.DataFrame(config_dict)

df["Set"] = df.name.str.split("<br>").apply(lambda x: x[0])
df["Architecture"] = df.name.str.split("<br>").apply(lambda x: ''.join(x[1].split(" ")[:-1]))
df["Run"] = df.name.str.split("<br>").apply(lambda x: x[1].split(" ")[-1]).astype(int)
df["Type"] = "DGPS"
df.Type[df.Run > 99] = "stratified"
df.drop(columns=["name", "type"], inplace=True)
df = df.explode(["x", "y"]).reset_index(drop=True)
df = df.replace({"f1":"training", "val_f1":"validation"})
df.head()

plt.figure()
sns.lineplot(x="x", y="y", data=df, hue="Architecture", 
             style="Set", style_order=["training","validation"])
plt.yticks(np.arange(0.7,1.01,0.05))
plt.ylabel("F1")
plt.xlabel("Epoch")
plt.xlim(0)
plt.ylim(0.7)
plt.savefig("f1_vad_gn.png", dpi=600, bbox_inches="tight")
plt.show(block=False)
# %%
