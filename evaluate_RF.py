#%%
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math
import scipy.signal as sig
import pandas as pd
import json
import seaborn as sns   
import pickle
from glob import glob

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
            
from tqdm import tqdm
from joblib import Parallel, delayed
tot_data, name = [], []

def get_results(f):
    import warnings
    warnings.simplefilter("ignore")
    warnings.filterwarnings('ignore')
    print(f)
    thresholds=[.37,2]
    #thresholds=[20]
    with open(f, "rb") as data:
        results = pickle.load(data)
    data_new, _ = test_evaluation(results, inSamples=False, thresholds=thresholds, without_axle=11)

    df = pd.DataFrame(data = data_new, columns=['Grenzwert in m', 'Überfahrt', 'Sensor', 'n Achsen', 'korrekte Achsen', 
                                                'vorhergesagte Achsen', 'Geschwindigkeiten', 'Abweichung Samples', 'Abweichung Meter', 
                                                'F1', 'Precision', 'Recall', 'Durschnitt cm', 'Durschnitt Sample', 'AxId', 'AccPerAxle'])

    data = []
    for t in thresholds:
        for s in [7,11]:            
            _df = df[df["Sensor"] != s]
            _df = _df[df['Grenzwert in m']==t]
            tp = _df['korrekte Achsen'].sum()
            fp = (_df['vorhergesagte Achsen'].sum()-tp)
            fn = (_df['n Achsen'].sum()-tp)
            f1 = tp/(tp+fp/2+fn/2)
            data.append([t,s,f1,f,_df['Abweichung Samples'],_df['Abweichung Meter'],_df['Sensor'], _df['F1']])
            
    return data

#for f in tqdm(glob("rf/*.bin")):
#    get_results(f)

if __name__ == '__main__':
    results_tot = Parallel(n_jobs=25)(delayed(get_results)(f) for f in tqdm(glob("final/*.bin")))

print(results_tot)

import pickle
with open('RFresults_final.bin', "wb") as f:
    pickle.dump(results_tot, f)
#%%
