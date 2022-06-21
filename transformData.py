#%%
import numpy as np
from joblib import Parallel, delayed
import numpy as np
import io, os, pywt, zipfile
import utils.utils as ut
import matplotlib.pyplot as plt

def cropDataTables(X, y):
    label = np.zeros_like(X)
    for i in range(y.shape[1]):
        label[y[:,i],i] = 1  
    example = X[y.min()-150:y.max()+500]
    label = label[y.min()-150:y.max()+500]
    return example, label.astype(int)
    
names = [["acc_li_", "idxl_acc"], ["acc_re_", "idxr_acc"]]

def compressData(p):
    documentation = []
    with zipfile.ZipFile(PATH) as zipper:
        x = []
        y = []
        try:
            for n in names: 
                with io.BufferedReader(zipper.open(f"{FOLDER}/{n[0]}{p}.txt", mode='r')) as f:
                # with io.BufferedReader(zipper.open(f"New Data - BIM2/{n[0]}{p}.txt", mode='r')) as f:
                # with io.BufferedReader(zipper.open(f"{n[0]}{p}.txt", mode='r')) as f:
                    x.append(np.genfromtxt(f, delimiter = ','))
                with io.BufferedReader(zipper.open(f"{FOLDER}/{n[1]}{p}.txt", mode='r')) as f:
                    y.append(np.genfromtxt(f, delimiter = ','))
                documentation.append(f"{n[0]}{p} | {x[-1].shape} | {n[1]}{p} | {y[-1].shape}")
            x = np.hstack(x)
            y = np.hstack(y)       
            x, y = cropDataTables(x, y.astype(int))   
            documentation.append(f" ")
            documentation.append(f"{n[0]}{p} | {x.shape} | {n[1]}{p} | {y.shape}")
            documentation.append(f" ")
            documentation.append(f" ")
            np.savez(f"{BASE}/cropped/{TARGET}/{p}.npz", x=x, y=y)
        except:
            pass
    with open(f"{BASE}/documentation {TARGET}.txt", 'w') as output:
        for row in documentation:
            output.write(row + '\n')

def getWavelet(signal, set, fs):
    scales = np.linspace(set[1],set[2],16)
    [coefficients, _] = pywt.cwt(signal, scales, set[0], 1/fs)
    temp = abs(coefficients).T 
    return temp/temp.max()   
                     
def toModelInput(name, config):
    data = np.load(f'{BASE}/cropped/{FOLDER}/{name}')
    X = data['x']
    y = data['y']
    transforms = []
    for i, _x in enumerate(X.T):        
        s = []
        _key = "acc" #if i < 10 else "str"
        c = config[_key]           
        
        for set in c['wl']:
            s.append(getWavelet(_x, set, config["fs"]))
            
        s = np.dstack(s).astype(np.float32)
        transforms.append(s) 
    transforms = np.stack(transforms, axis=3).astype(np.float32)
    np.savez(f"{BASE}/transformed/{TARGET}/{name}", x=transforms, y=y, allow_pickle=True)


#['cgau1','cgau2','cmor','gaus2','mexh','morl','shan']
def plotWL(time, signal, scales, waveletname = 'cgau1', cmap = 'magma', 
                 ylabel = 'Frequencie in Hz', xlabel = 'Time in s', fs = 600,
                 contour = True, axle = None, name=None):
    
    coefficients, frequencies = pywt.cwt(signal, scales, waveletname, 1/fs)
    pwr = abs(coefficients)

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.contourf(time, np.log2(frequencies), pwr, 255, extend='both',cmap=cmap)

    if isinstance(axle, np.ndarray):
        axle = np.argwhere(axle==1)
        for axl in axle:
            ax.plot([axl / fs, axl / fs], [np.log2(frequencies).min(), np.log2(frequencies).max()], linestyle=(0, (1, 15)), color='c')
            
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    
    yticks = 2**np.append(np.arange(np.ceil(np.log2(frequencies.min())), np.ceil(np.log2(frequencies.max()))),np.log2(frequencies.max()))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks.astype(int))

    fig.tight_layout()
    plt.savefig(f"{waveletname}{name}.png", dpi=300)
    plt.show()
                
def plotSettings(test_X, test_y, config, name=None):
    t = np.arange(0,len(test_X))/config["fs"]
    for i, let in enumerate(config["acc"]["wl"]):
        scales = np.linspace(let[1],let[2],256)
        plotWL(t,test_X,scales,waveletname=let[0] ,axle=test_y,contour=False,fs=config["fs"],name=i)

def testSettings():
    json_file = "../data_loader/transformConfigs/config_trans.json"
    _, config_dict, _ = ut.get_config_from_json(json_file)
    data = np.load(f"../data/10.npz")
    x = data['x']
    y = data['y']
    for i, (_x, _y) in enumerate(zip(x.T[:10],y.T[:10])):
        print(i)
        plotSettings(_x,_y,config_dict)
    
        fig, ax = plt.subplots(figsize=(14, 3))  
        ax.plot(np.arange(_x.size) / config_dict["fs"], _x, c='indigo', lw=0.5)
        ax.set_ylabel(r'Acceleration in $\frac{m}{s^2}$', fontsize=16)
        ax.set_xlabel('Time in s', fontsize=16)
        if isinstance(_y, np.ndarray):
            _y = np.argwhere(_y==1)
            for axl in _y:
                ax.plot([axl / config_dict["fs"], axl / config_dict["fs"]], [_x.min(), _x.max()], linestyle=(0, (1, 15)), color='c')
        fig.tight_layout()
        plt.savefig("signal.png", dpi=300)
        break


# testSettings()
#%%      
if __name__ == "__main__":        
    
    json_file = "../data_loader/transformConfigs/config_trans.json"
    _, config_dict, _ = ut.get_config_from_json(json_file)
    
    TARGET = config_dict["target"]
    FOLDER = config_dict["folder"]
    BASE = "E:/Riedel"
    PATH = f"{BASE}/cropped/{FOLDER}.zip"
    os.makedirs(f"{BASE}/cropped/{TARGET}", exist_ok=True)
    os.makedirs(f"{BASE}/transformed/{TARGET}", exist_ok=True)
    

    if config_dict["crop"]:
        Parallel(n_jobs=config_dict["n_jobs"])(delayed(compressData)(p) for p in range(1,config_dict["range"])) 
    
    names = os.listdir(f'{BASE}/cropped/{FOLDER}') 
    
    Parallel(n_jobs=config_dict["n_jobs"])(delayed(toModelInput)(name,config_dict) for name in names)

    