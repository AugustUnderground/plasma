"""
Modeling and Approximation the Operating Point Data of Primitive Devices.
    Requirements:
        - PREDICT data as hdf5, should be stored like so

        ```python
            import hdf5 as h5
            with h5_file h5.File("/path/to/{tech}-{device-type}.h5", "w"):
                for col in data:
                    h5_file[col] = data[col].to_numpy()
        ```

        where `data` is the PREDICT dataframe in python. (assuming pandas is
        used).

        - Dependencies installed:

        ```bash
        $ pip install -r ./requirements.text
        ```
"""

import os, datetime, time, sys
from expression import compose, pipe
from functools import partial, reduce
from itertools import product, repeat
from fastprogress.fastprogress import master_bar, progress_bar
import h5py as h5
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import joblib as jl
import torch as pt
import torch_optimizer as optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.model_selection._split import train_test_split

## Setup
rng_seed: int       = 666
eng                 = mpl.ticker.EngFormatter()
sns.set_theme(style = "darkgrid")

## File System and Data Loading
device_type: str = "nmos"
tech: str        = "gpdk180"
data_path: str   = f"../data/{tech}-{device_type}.h5"
time_stamp: str  = "20211210-172020"
model_dir: str   = f"./models/{time_stamp}-{device_type}-{tech}"

params_x: [str] = ["Vds", "Vgs", "Vbs", "W", "L"]
params_y: [str] = [ "vth", "vdsat", "ron", "gm", "gmbs", "gds"
                  , "betaeff", "id", "vearly", "pwr", "fug" ]

normed_y: [str] = [ "ron","gm","gmbs","gds","betaeff","id","pwr"]
trafo_y: [str]  = [ "fug", "betaeff", "pwr", "gmbs", "gds", "gm", "vdsat"
                  , "ron", "id", "vearly"]

## Evaluation
## Reload the trained model for a fresh start, as you would in production.

class PlasmaModel:
    """
    Prediction Model for a single Primitive Device (NMOS/PMOS) in a specific
    Technology.
    """
    def __init__( self, path: str, params_x: [str], params_y: [str]
                , trafo_x: [str], trafo_y: [str], norm_y: [str] ):
        self.params_x = params_x
        self.params_y = params_y
        self.mask_x   = np.array([int(px in trafo_x) for px in self.params_x])
        self.mask_y   = np.array([int(py in trafo_y) for py in self.params_y])
        self.norm_y   = norm_y
        self.scaler_x = jl.load(f"{path}/scale.X")
        self.scaler_y = jl.load(f"{path}/scale.Y")
        self.scale_x  = lambda X: self.scaler_x.transform(X)
        self.scale_y  = lambda Y: self.scaler_y.inverse_transform(Y)
        self.trafo_x  = lambda X: ( ( np.log10( np.abs(X)
                                              , where = (np.abs(X) > 0)
                                              ) * self.mask_x)
                                  + (X * (1 - self.mask_x)))
        self.trafo_y  = lambda Y: ( (np.power(10, Y) * self.mask_y) 
                                  + (Y * (1 - self.mask_y)))
        self.model    = pt.jit.load(f"{path}/model.pt").cpu().eval()
    def _predict(self, X: np.array) -> np.array:
        with pt.no_grad():
            return pipe( X
                       , self.trafo_x
                       , self.scale_x
                       , np.float32
                       , pt.from_numpy
                       , self.model
                       , pt.Tensor.numpy
                       , self.scale_y
                       , self.trafo_y
                       , )
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Y = pd.DataFrame( self._predict(X[self.params_x].values)
                        , columns = self.params_y )
        Y[self.norm_y] = Y[self.norm_y].multiply( X.reset_index(drop = True)["W"]
                                                , axis = "index" )
        return Y
    def scalar_predict( self, vds: float, vgs: float, vbs: float
                         , w: float, l: float) -> pd.DataFrame:
        X = pd.DataFrame([[vds,vgs,vbs,w,l]], columns = self.params_x)
        Y = pd.DataFrame( self._predict(X.values), columns = self.params_y )
        Y[self.norm_y] = Y[self.norm_y].multiply( X.reset_index(drop = True)["W"]
                                                , axis = "index" )
        return Y
    def tensor_predict( self, vds: pt.Tensor, vgs: pt.Tensor, vbs: pt.Tensor
                         , w: pt.Tensor, l: pt.Tensor) -> pd.DataFrame:
        foo = np.array([1,2])
        return foo

mdl = PlasmaModel(model_dir, params_x, params_y, [], trafo_y, normed_y)

with h5.File(data_path, 'r') as db:
    col_names   = list(db.keys())
    data_matrix = np.transpose(np.array([db[c] for c in col_names]))
    data_raw    = pd.DataFrame(data_matrix, columns=col_names).dropna()

## EVALUATION SPEED

num_points  = 500000
batch_sizes = list(map(lambda x: 5 * (10 ** x), range(6)))[2:]

def eval_time(x: int, y: int, b) -> float:
    tru_x = traces[params_x][x:y]
    tru_y = traces[params_y][x:y]
    tic   = time.time()
    prd_y = mdl.predict(tru_x)
    toc   = time.time()
    t     = toc - tic
    b.comment = f"Eval Time per Batch: {t:.4}s"
    return t

def eval_model(b: int, n: int) -> float:
    print(f"Evaluating {num_points} with {b} points per batch.")
    traces  = data_raw.sample(n, replace = False)
    idx_bar = progress_bar(list(zip( np.arange(0,n,b).tolist()
                                   , np.arange(b,(n+b),b).tolist())))
    ot      = np.array([eval_time(x,y,idx_bar) for (x,y) in idx_bar]).sum()
    print(f"\nOverall Evaluation {ot:.4}s.\n")
    return ot

averages = [eval_model(b,num_points) for b in batch_sizes]

#fig, (ax1, ax2) = plt.subplots(figsize=(14,7), nrows=1, ncols=2)
fig, ax1 = plt.subplots(figsize=(14,7), nrows=1, ncols=1)
num_batches = [num_points / b for b in batch_sizes]
ax1.plot(num_batches,averages)
ax1.set_title(f"Evaluation Time for different Batch Sizes.")
ax1.set_xlabel("# Batches")
ax1.set_ylabel("Time [s]")
plt.show()

## PARTIAL EVALUATION

Vgs = 0.9
Vds = 0.9
Vbs = 0.0

traces = data_raw[params_x + params_y].copy().reset_index(drop=True)
traces["Vgs"] = round(traces.Vgs, ndigits = 2)
traces["Vds"] = round(traces.Vds, ndigits = 2)
traces["Vbs"] = round(traces.Vbs, ndigits = 2)

widths  = traces.W.unique().tolist()
lenghts = traces.L.unique().tolist()

trace = traces[(traces.Vds == Vds) & (traces.Vgs == Vgs) & (traces.Vbs == Vbs)]
sweep = list(product(widths, lenghts))
pmdl  = partial(mdl.scalar_predict, Vds, Vgs, Vbs)

tic  = time.time()
pres = pd.concat([pmdl(w,l) for (w,l) in sweep])
toc  = time.time()
t    = toc - tic
print(f"Partially Applied Evaluation took {t:.4}s.")

inp  = pd.concat([pd.DataFrame([[Vds, Vgs, Vbs, w,l]], columns = params_x) for (w,l) in sweep])
tic  = time.time()
dres = mdl.predict(inp)
toc  = time.time()
t    = toc - tic
print(f"Evaluation took {t:.4}s.")
