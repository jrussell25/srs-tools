import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from tqdm import tqdm


def check_consecutive(x: pd.Series, N: int = 15) -> pd.Series:
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


def exp_approach(x: np.ndarray, a: float, k: float) -> np.ndarray:
    return a * (1 - np.exp(-k * x))


def exp_approach2(x: np.ndarray, a: float, k: float, t: float) -> np.ndarray:
    return a * (1 - np.exp(-k * (x - t)))


def fit_exponential(
    x: np.ndarray, y: np.ndarray, p0: tuple[float, float]
) -> np.ndarray:
    try:
        p, cov = curve_fit(
            exp_approach, x, y, p0=p0, bounds=(0, np.inf), absolute_sigma=True
        )
        out = np.array([*p, cov[0, 0], cov[1, 1]])
    except RuntimeError:
        out = np.array([np.nan, np.nan, np.nan, np.nan])
    return out


def fit_all_exponentials(
    df: pd.DataFrame, t: np.ndarray, p0: tuple[float, float], prog_bar: bool = True
) -> pd.DataFrame:
    all_params = pd.DataFrame(
        index=df.index, columns=["a", "k", "sig_a", "sig_k"], dtype="f4"
    )

    if prog_bar:
        iterator = tqdm(df.iterrows(), total=len(df))
    else:
        iterator = df.iterrows()

    for i, (idx, s) in enumerate(iterator):
        y = s.values
        mask = ~np.isnan(y)
        y = y[mask]
        x = t[mask]
        all_params.loc[idx] = fit_exponential(x, y, p0)

    return all_params


def trace_lineages(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Indentify individual mother daughter pairs and then identify
    the lineage i.e. oldest observed ancestor of each cell.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Dataframe containing center of mass info (ideally from nuclear localizations)
        for cells in **in a single FOV**. Should have "T" and "CellID" in index.
        See Note below for applying to multiple FOVs

    Returns
    -------
    lineage_info : pd.DataFrame

    Note
    ----
    It is a nice oneliner to do this for all FOVs in a dataset:
    pd.concat({s:trace_lineages(df.loc[s]) for s in df.index.unique("S")}, names=["S"])
    """
    first_frames = tracks_df.reset_index("T").groupby("CellID")["T"].min()

    row_ind = first_frames.loc[first_frames == 0].index.tolist()
    col_ind = first_frames.loc[first_frames == 0].index.tolist()
    for t, s in first_frames.groupby(first_frames):
        if t > 0:
            ftracks = tracks_df.loc[t]
            new_ids = s.index
            new = ftracks.loc[new_ids]
            old = ftracks.drop(new_ids)

            dists, candidate_idx = KDTree(old.values).query(
                new.values, distance_upper_bound=25
            )
            for d, m in zip(new.index.values, candidate_idx):
                row_ind.append(d)
                if m < len(old):
                    col_ind.append(m)
                else:  # map anonymous cells to themselves
                    col_ind.append(d)

    N = first_frames.index.max() + 1  # values.shape[0]
    adj = csr_array(
        (np.ones((len(row_ind),), dtype="u2"), (row_ind, col_ind)), shape=(N, N)
    )
    n, components = connected_components(adj)
    return pd.DataFrame(
        {"CellID": row_ind, "mother": col_ind, "lineage": components[row_ind]}
    ).set_index("CellID")
