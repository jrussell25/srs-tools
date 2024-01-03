import numpy as np
import pandas as pd


class UptakeExperiment:
    def __init__(
        self,
        N_init: int,
        N_gen: int,
        tau: float,
        sigma_div: float,
        f_exchange: float,
        dt: float,
        sigma_y: float,
        seed: int | None = None,
    ):
        self.N_init = N_init
        self.N_gen = N_gen
        self.tau = tau
        self.sigma_div = sigma_div
        self.sigma_y = sigma_y

        self.rng = np.random.default_rng(seed)

        self.cell_table = self._build_cell_lineages()
        self.trajectories = pd.DataFrame()

    def _build_cell_lineages(self) -> pd.DataFrame:
        return pd.DataFrame()

    def _build_trajectories(self) -> pd.DataFrame:
        return pd.DataFrame()
