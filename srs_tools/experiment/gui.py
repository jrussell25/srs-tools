import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display


class MicroscopyProgbars:
    def __init__(self, dims, sizes):
        self.sizes = pd.Series(sizes, index=dims, dtype="uint16")
        self.count = pd.Series(np.zeros(len(dims)), index=dims, dtype="uint16")

        self.pbar_vbox = self._setup_pbars()

    def _setup_pbars(self):
        pbars = []
        for i, (d, s) in enumerate(self.sizes.items()):
            pb = widgets.IntProgress(min=0, max=s, value=0, description=d)
            label = widgets.Label(value=f"0/{s}")
            pbars.append(widgets.HBox([pb, label]))
        pbars = widgets.VBox(pbars)
        return pbars

    def _ipython_display_(self):
        display(self.pbar_vbox)

    def update_count(self, frame_count):

        full = frame_count
        for d, s in self.sizes.iloc[::-1].items():
            self.count[d] = full % s
            full = full // s

    def update_pbars(self, frame_count):
        self.update_count(frame_count)
        for i, (d, s) in enumerate(self.count.items()):
            self.pbar_vbox.children[i].children[0].value = s
            self.pbar_vbox.children[i].children[1].value = f"{s}/{self.sizes[d]}"
