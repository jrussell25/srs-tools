from pathlib import Path
from typing import Any

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
        lazy: bool = False,
    ):
        self.N_init = N_init
        self.N_gen = N_gen
        self.tau = tau
        self.sigma_div = sigma_div
        self.sigma_y = sigma_y
        self.f_exchange = f_exchange
        self.dt = dt

        self.rng = np.random.default_rng(seed)

        if not lazy:
            self.cell_table = self._build_cell_lineages()
            self.trajectories = self._build_trajectories()

    def _build_cell_lineages(self) -> pd.DataFrame:
        cells: list[tuple[Any, ...]] = []
        for n in range(self.N_init):
            v0 = self.rng.uniform()
            lineage = [(len(cells), -1, -v0, n)]
            for c in lineage:
                idx, mother, tb, lng = c
                for g in range(self.N_gen + 1):
                    td_deterministic = tb + np.log(2) * (1 + g)
                    td = np.clip(
                        td_deterministic + self.sigma_div * self.rng.normal(), 0, None
                    )
                    if td < self.N_gen:
                        lineage.append((len(cells) + len(lineage), idx, td, n))
            cells = [*cells, *lineage]
        return pd.DataFrame(
            cells, columns=["idx", "mother", "tb", "lineage"]
        ).set_index("idx")

    def _build_trajectories(self) -> pd.DataFrame:
        nT = int(self.N_gen / self.dt) + 1
        t = np.linspace(0, self.N_gen, num=nT)
        trajectories = pd.DataFrame(index=self.cell_table.index, columns=range(nT))
        traj0 = self.f_exchange * (1 - np.exp(-t / self.tau))
        for i in self.cell_table.index:
            m = self.cell_table.loc[i, "mother"]
            tb = self.cell_table.loc[i, "tb"]
            if m == -1:
                trajectories.loc[i] = traj0
            else:
                t_idx = np.argmin(np.abs(t - tb))
                tloc = t[t_idx:] - t[t_idx]
                m_val = trajectories.loc[m, t_idx]
                val_max = (1 - self.f_exchange) * m_val + self.f_exchange
                trajectories.loc[i, t_idx:] = val_max - (val_max - m_val) * np.exp(
                    -tloc / self.tau
                )

        trajectories += self.rng.normal(scale=self.sigma_y, size=trajectories.shape)
        return trajectories

    def save(self, filepath: str | Path) -> None:
        raise NotImplementedError


def load(filepath: str | Path) -> UptakeExperiment:
    raise NotImplementedError
    config = pd.read_hdf(filepath, key="config")
    cell_table = pd.read_hdf(filepath, key="cells")
    trajectories = pd.read_hdf(filepath, key="trajectories")
