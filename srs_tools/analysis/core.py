import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from scipy.optimize import curve_fit

def check_consecutive(x:pd.Series, N:int=15)->pd.Series:
    """
    Generate a binary mask indicating the cells and timepoints which are part of a
    stretch of N consecutive observations of that cell. Intended for use with a groupby
    operation on a full dataset.

    Parameters
    ---------- 
    x: pd.Series
        Series containing a single cell's data with nan where that cell is not
        observed. 
    N: int
        Number of consecutive observations to check for.

    Returns
    ------- 
    mask: pd.Series
        Boolean series aligned with x indicating the time points that comprise N or more
        consecutive observations.
    """

    m = x.isna()
    s = m.cumsum()
    final_mask = s.map(s[~m].value_counts()).ge(N) & ~m
    return final_mask


def exp_approach(x:np.ndarray, a:float, k:float)->np.ndarray:
        return a*(1-np.exp(-k*x))

def exp_approach2(x:np.ndarray, a:float, k:float, t:float)->np.ndarray:
        return a*(1-np.exp(-k*(x-t)))

def fit_exponential(x:np.ndarray, y:np.ndarray, p0:tuple[float, float])->np.ndarray:
    try:
        p, cov = curve_fit(exp_approach, x, y, p0=p0, bounds=(0,np.inf), absolute_sigma=True)
        out = np.array([*p, cov[0,0], cov[1,1]])
    except RuntimeError:
        out = np.array([np.nan, np.nan, np.nan, np.nan])
    return out

def fit_all_exponentials(df:pd.DataFrame, t:np.ndarray, p0:tuple[float, float], prog_bar=True)-> pd.DataFrame:

    all_params = pd.DataFrame(index=df.index, columns=['a','k', 'sig_a', 'sig_k'], dtype='f4')

    if prog_bar:
        iterator = tqdm(df.iterrows(), total=len(df))
    else:
        iterator = df.iterrows()

    for i, (idx, s) in enumerate(iterator):
        y = s.values
        mask = ~np.isnan(y)
        y = y[mask]
        x = t[mask]
        all_params.loc[idx] = fit_exponential(x,y,p0)
    
    return all_params

