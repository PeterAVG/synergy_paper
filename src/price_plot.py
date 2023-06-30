#%%
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.loader import DataLoader
from src.utils import _set_font_size

sns.set_theme()
sns.set(font_scale=1.5)


data_loader = DataLoader()

spot_prices = data_loader._lambda_spot.reshape(-1)  # type:ignore
mfrr_prices = data_loader._lambda_mfrr.reshape(-1)  # type:ignore
balance_prices = data_loader._lambda_rp.reshape(-1)  # type:ignore
_hours = list(range(1, len(spot_prices) + 1))

hours = [
    datetime.datetime(2022, 1, 1) + datetime.timedelta(hours=h)  # type:ignore
    for h in _hours
]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# lineplots of prices
ax.step(
    hours,
    spot_prices,
    label=r"$\lambda_{h}^{s}$",
    color="red",
    where="post",
    alpha=0.7,
)
ax.step(
    hours,
    balance_prices,
    label=r"$\lambda_{h}^{b}$",
    color="orange",
    where="post",
    alpha=0.5,
)
ax.step(
    hours,
    mfrr_prices,
    label=r"$\lambda_{h}^{mFRR}$",
    color="blue",
    where="post",
)

ax.set_ylabel("Price [DKK/kWh]")
ax.legend(loc="best")
ax.xaxis.set_tick_params(rotation=45)
_set_font_size(ax, legend=16)
plt.tight_layout()

plt.savefig("tex/figures/prices.png", bbox_inches="tight", dpi=300)
