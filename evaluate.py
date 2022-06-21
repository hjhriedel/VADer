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

config = process_config("configs/axle_05.json")
model = fn.Model(config,(16,6)).model
PATH = "experiments/2022-06-21/m00-01-dl04-checkpoints/17-53/"
model.load_weights(PATH)
test_names = np.load(PATH + "test_names.ob", allow_pickle=True)
print(test_names)
vPATH = "E:/Riedel/v" # speed data
dPATH = "E:/Riedel/transformed/Data TrigG1 new Code/" # acceleration data

def loadRaw(PATH = dPATH):
    data = np.load(PATH, allow_pickle=True)
    x, y = data['x'], data['y']
    data.close
    width = math.ceil(len(y)/16)*16
    x_padded, y_padded = np.zeros((width,x.shape[1],x.shape[2],x.shape[3])), np.zeros((width,x.shape[3]))
    x_padded[:len(y)], y_padded[:len(y)] = x, y
    return x_padded[:,:,4:, :10], y_padded

def plot_results(pred, peaks, tv, plot):
    t = np.arange(len(pred.flatten()))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, pred.flatten(),zorder=2, lw=0.5)
    ax.plot([t[tv>0],t[tv>0]], [0,1], color='r',zorder=0, lw=0.5)
    ax.plot([peaks], [pred.flatten()[peaks]], 'v', color='m',zorder=0, lw=0.5)
    plt.xlabel(f'Time step')
    plt.xlim([0, 2000])
    plt.ylabel(f'Prediction')
    plt.tight_layout()
    plt.savefig(f"{plot[2]}_{plot[1]}.png",dpi=400)
    plt.clf()

def plot_json():
    json_file = "../val_precision_vs_val_recall_chart_data.json"
    with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)
    df = pd.DataFrame(config_dict)
    ax = sns.scatterplot(x="recall", y="precision", data=df.iloc[1:-2], palette="rocket_r",s= 80, hue="gamma", ci=None)
    plt.savefig('gamma.png',dpi=600)

def error_axle_positions(prediction, true_values, v, boundary=0.4, distance=1.0, plot=[False], boundaryInSamples=True):

    # defining number and position of true axles:
    idx_axle_true = np.where(true_values == 1)[0]
    num_axle_true = np.count_nonzero(true_values)
    distanceInSample = distance if boundaryInSamples else math.floor(distance / np.mean(v) * 600)

    # defining number and position of predicted axles:
    idx_axle_pred, _ = sig.find_peaks(prediction, height=0.25, prominence=0.15, distance=distanceInSample)
    num_axle_pred = idx_axle_pred.shape[0]
    if plot[0]:
        plot_results(prediction, idx_axle_pred, true_values, plot)

    num_axle_correct, num_axle_false = 0, 0
    false_samples, false_m = np.zeros_like(idx_axle_true), np.zeros_like(idx_axle_true, dtype=np.float64)
    axle_correct = np.zeros_like(true_values)
    for i, _v in zip(range(num_axle_true),v):
        if idx_axle_pred.any():
            idx_min_distance = np.argmin(abs(idx_axle_pred - idx_axle_true[i]))
            boundaryInSample = boundary if boundaryInSamples else boundary / _v * 600
            if min(abs(idx_axle_pred - idx_axle_true[i])) <= boundaryInSample:
                false_samples[i-num_axle_false] = idx_axle_true[i] - idx_axle_pred[idx_min_distance]
                false_m[i-num_axle_false] = (idx_axle_true[i] - idx_axle_pred[idx_min_distance]) / 600 * _v
                num_axle_correct += 1
                idx_axle_pred = np.delete(idx_axle_pred, idx_min_distance)
                axle_correct[idx_axle_true[i]] = 1
            else:
                false_samples = np.delete(false_samples, i-num_axle_false, axis=0)
                false_m = np.delete(false_m, i-num_axle_false, axis=0)
                num_axle_false += 1
        else:
            false_samples = np.delete(false_samples, i - num_axle_false, axis=0)
            false_m = np.delete(false_m, i - num_axle_false, axis=0)
            num_axle_false += 1
    return num_axle_true, num_axle_pred, num_axle_correct, false_samples, false_m

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
    boundaries = np.hstack(np.array([[2.0,0.5,0.35,0.2],]*(len(recall[0]))).T)
    df['boundary']=np.hstack((boundaries,boundaries))
    fig, ax = plt.subplots(1, 1, figsize=[8,5])
    ax = sns.violinplot(x="boundary", y='value', hue='metric',
                        data=df, split=True, inner="quartile", cut=0, scale="count")
    plt.xlabel(f'Threshold in meter')
    plt.ylabel(f'Score')
    ax.set_ylim(0,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
    ax.spines['bottom'].set_visible(False)     
    plt.tick_params(bottom = False) 
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('pr.png',dpi=600)
    
    
def plot_norm_disp(dir, false_samples, name="", unit='samples'):
    mean = np.mean(false_samples)
    std = np.std(false_samples)
    h = sorted(np.arange(mean-std*4,mean+std*4,0.5))
    normal = stats.norm.pdf(h, mean, std)
    plt.figure(1)
    # plt.plot(h,normal) 
    plt.hist(false_samples, bins=39, density=True)
    plt.xlabel(f'Deviation in {unit}')
    plt.ylabel(f'Density')
    plt.savefig(f'{name}{dir}_norm.png', dpi=300)
    plt.clf()

#%%
dPATH = "E:/Riedel/transformed/Data TrigG1 new Code/"
def test_evaluation(names, model=model, inSamples=False):
    # inSamples=False
    precision, recall, f1, samples, meter = [], [], [], [], []
    for boundary in [2.0, 0.5, 0.35, 0.2]:
        fs, fm, all_stats = [], [], []
        for name in test_names:
            measurement, label = loadRaw(PATH = dPATH + name)  
            vl = np.vstack(np.genfromtxt(f"{vPATH}/v_idxl_acc{name[:-4]}.txt", delimiter=","))
            vr = np.vstack(np.genfromtxt(f"{vPATH}/v_idxr_acc{name[:-4]}.txt", delimiter=","))
            V = np.hstack((vl,vr)) # axle speed in m/s
            num_axle_true, num_axle_pred, num_axle_correct, false_samples, false_m, _stats = [], [], [], [], [], []
            i = 0
            for m, true_values, v in zip(measurement.T, label.T, V.T):
                prediction = model.predict(np.array([m.T]))[0]
                # acc_list, mean_fs_list, std_fs_list, avg_neg_list, avg_pos_list, percent_null_list, percent_neg_list, percent_pos_list, recall_list = [], [], [], [], [], [], [], [], []
                # boundary = 20 if inSamples else 1.0 # false_m in meter from exact position
                distance = 20 if inSamples else 2.0 # distance in meter of axles inbetween a bogie
                _num_axle_true, _num_axle_pred, _num_axle_correct, _fs, _false_m = error_axle_positions(prediction, true_values.T, v, boundary=boundary, distance=distance, plot=[False,i,name[:-4]], boundaryInSamples=inSamples)
                num_axle_true.append(_num_axle_true)
                num_axle_pred.append(_num_axle_pred)
                num_axle_correct.append(_num_axle_correct)
                _stats.append(error_statistics_2(_num_axle_true, _num_axle_pred, _num_axle_correct))
                false_samples.append(_fs)                                    
                false_m.append(_false_m)        
                i+=1
            false_samples, false_m = np.hstack(false_samples), np.hstack(false_m)
            fs.append(false_samples)           
            fm.append(false_m)           
            all_stats.extend(_stats)
        fs, fm = np.hstack(fs), np.hstack(fm)
        plot_norm_disp(f'total{inSamples}{boundary}', fs)
        plot_norm_disp(f'total{inSamples}{boundary}', fm*100, name="dv", unit='cm')  
        _f1, pre, rec = zip(*all_stats)
        precision.append(pre)
        recall.append(rec)
        f1.append(_f1)
        samples.append(fs)
        meter.append(fm)
        print(boundary)
        print('mean: ', np.mean(abs(fm)), ' std: ', np.std(fm), 
        ' pre: ', np.mean(precision), ' rec: ', np.mean(recall),
        ' f1: ', np.mean(f1))   
        
    violinplot(precision, recall)
    
    
test_evaluation(test_names)