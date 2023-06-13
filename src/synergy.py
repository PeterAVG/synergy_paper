#%%

import numpy as np

from src.loader import DataLoader

data_loader = DataLoader()

N = 1000
BURN_IN = 10
M = data_loader.M - BURN_IN
M = np.minimum(M, 40)

experiments = {}

for n in range(2, N + 1):

    print(f"Running experiment {n} of {N}...", end="\r")

    history = {
        "lambda_spot": [],
        "lambda_reserve": [],
        "lambda_balancing": [],
        "power": [],
        "profit_individual": [],
        "Reserve profit individual": [],
        "profit_aggregated": [],
        "Reserve profit aggregated": [],
    }

    rng = np.random.default_rng(24)
    lambda_pen = 0.1

    # simulate over a horizon of M days
    for m in range(M + BURN_IN):

        # get N uniform integer between 0 and 23
        x = rng.integers(0, 24, n)
        # NOTE: x represent any assets during a day. Could also be a stochastic process...

        # set values in placeholder to 1 for each row corresponding to index in x
        power = np.zeros((n, 24))
        power[np.arange(n), x] = 1

        # power now encodes when, e.g., a ventilation system is ON and OFF during the day.

        # prices
        # lambda_reserve = np.ones(24) * 0.1
        # lambda_spot = np.ones(24) * 0.5
        # lambda_balancing = lambda_spot + np.maximum(0, rng.uniform(-1.5, 1.5, 24))
        lambda_reserve = data_loader._lambda_mfrr[m, :]
        lambda_spot = data_loader._lambda_spot[m, :]
        lambda_balancing = data_loader._lambda_rp[m, :]

        # append prices to history
        history["lambda_spot"].append(lambda_spot)
        history["lambda_reserve"].append(lambda_reserve)
        history["lambda_balancing"].append(lambda_balancing)
        history["power"].append(power)

        if m < BURN_IN:
            continue

        # we now simulate the synergy effect of aggregating ventilation systems
        # versus simulating them individually

        # p_reserve is the expected power of the aggregated system and individual asset
        # based in previous day

        ### INDIVIDUAL ASSETS ###
        p_reserve = history["power"][-2]

        # activation happens when lambda_balancing > lambda_spot
        up_regulation = lambda_balancing > lambda_spot

        # calculate required activation of individual assets
        should_up_regulate = p_reserve * up_regulation

        # calculate actual up-regulation of individual assets (should match with actual power consumption)
        actual_up_regulation = should_up_regulate - np.maximum(
            0, should_up_regulate - power
        )

        # calculate actication profit
        profit_activation = actual_up_regulation * (lambda_balancing - lambda_spot)

        # calculate penalty for not delivering the required up-regulation
        penalty = np.maximum(0, should_up_regulate - actual_up_regulation) * lambda_pen

        # calculate reserve profit
        profit_reserve = p_reserve * lambda_reserve

        # total profit
        profit = profit_activation + profit_reserve - penalty

        history["profit_individual"].append(profit.sum())
        history["Reserve profit individual"].append(profit_reserve.sum())

        #####################################################################

        ### AGGREGATED ASSETS ###
        p_reserve_agg = history["power"][-2].sum(axis=0)

        # calculate required activation of portfolio
        should_up_regulate = p_reserve_agg * up_regulation

        # calculate actual up-regulation of portfolio (should match with actual power consumption)
        actual_up_regulation = should_up_regulate - np.maximum(
            0, should_up_regulate - power.sum(axis=0)
        )

        # calculate actication profit
        profit_activation = actual_up_regulation * (lambda_balancing - lambda_spot)

        # calculate penalty for not delivering the required up-regulation
        penalty = np.maximum(0, should_up_regulate - actual_up_regulation) * lambda_pen

        # calculate reserve profit
        profit_reserve = p_reserve_agg * lambda_reserve

        # total profit
        profit = profit_activation + profit_reserve - penalty

        history["profit_aggregated"].append(profit.sum())
        history["Reserve profit aggregated"].append(profit_reserve.sum())

    # print(f"Profit individual: {np.sum(history['profit_individual'])}")
    # print(f"Profit aggregated: {np.sum(history['profit_aggregated'])}")
    # print(f"Reserve profit individual: {np.sum(history['Reserve profit individual'])}")
    # print(f"Reserve profit aggregated: {np.sum(history['Reserve profit aggregated'])}")

    assert np.isclose(
        np.sum(history["Reserve profit individual"]),
        np.sum(history["Reserve profit aggregated"]),
    )

    experiments[n] = history


# Conclusion: of course there is a huge synergy effect because of the Law of Large Numbers and
# Central Limit Theorem. The aggregated system is much more predictable than the individual assets.

#%%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import _set_font_size

# sns.set_theme()
# sns.set(font_scale=1.5)


# plot histogram of x
f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax = ax.ravel()
# plot barplot for first asset
asset1 = np.zeros(24)
asset1[x[0]] = 1
ax[0].bar(np.arange(1, 25), asset1, edgecolor="black")
ax[1].hist(x, bins=24, edgecolor="black")
ax[0].set_xlabel("Hour")
ax[1].set_xlabel("Hour")
ax[0].set_ylabel("Power [kW]")

plt.savefig("tex/figures/assets.png", bbox_inches="tight", dpi=300)

# stacked bar plot of x
G = 5
groups = rng.integers(1, G + 1, N)
f, ax = plt.subplots(1, 1, figsize=(6, 5))
bottom = np.zeros(24)
# _n = 200
# assert _n < len(x)
# for j in range(_n):
colors = sns.color_palette("tab10", G)
for j in range(G + 1):
    ix = groups == j
    print(j, end="\r")
    _y = np.zeros(24)
    # count number of times each hour is activated from x[ix]
    for i in range(24):
        _y[i] = np.sum(x[ix] == i)
    # _y[x[j]] = 1
    # alpha = 0.7 if bottom[x[j]] % 2 == 0 else 0.3
    color = colors[j - 1]
    # ax.bar(x[j]+1, _y, edgecolor="black", bottom=bottom, color=color)
    ax.bar(
        np.arange(1, 25),
        _y,
        edgecolor="black",
        bottom=bottom,
        color=color,
        label=f"Group {j+1}",
        width=1.0,
        alpha=0.7,
    )
    # ax.bar(x[j]+1, _y, bottom=bottom, color="C0")
    bottom += _y.copy()
    # if j == 7:
    #     break

# plot hlines for each hour and y-integer if bottom < y
for i in range(24):
    for j in range(int(bottom.max())):
        if j < bottom[i]:
            ax.hlines(
                j,
                i + 1 - 0.5,
                i + 2 - 0.5,
                color="black",
                linestyle="-.",
                linewidth=1.0,
                alpha=0.3,
            )

ax.set_xlabel("Hour")
ax.set_xlim(0.5, 24.5)
# ax.set_ylim(0, 3)
ax.set_ylabel("Power [kW]")
ax.legend()
_set_font_size(ax, 16, 8)
plt.tight_layout()
plt.savefig("tex/figures/assets2.png", bbox_inches="tight", dpi=300)

sns.set_theme()
sns.set(font_scale=1.5)

# plot of x vs profits for each experiment
f, ax = plt.subplots(1, 1, figsize=(10, 5))
xrange = list(experiments.keys())
asset_profit = [np.sum(experiments[n]["profit_individual"]) for n in xrange]
agg_profit = [np.sum(experiments[n]["profit_aggregated"]) for n in xrange]
# ax.plot(xrange, asset_profit, label="Individual assets")
# ax.plot(xrange, agg_profit, label="Aggregated assets")
ax.plot(xrange, np.array(agg_profit) / np.array(asset_profit), label="Synergy effect")
ax.plot(
    xrange,
    pd.Series(np.array(agg_profit) / np.array(asset_profit)).rolling(30).mean().values,
    label="Synergy effect (rolling mean)",
)
ax.set_xlabel("# of assets")
ax.set_ylabel("Synergy effect")
ax.legend()

_set_font_size(ax, 16, 20)

plt.tight_layout()

# save to tex/figures folder
plt.savefig("tex/figures/synergy_effect.png", bbox_inches="tight", dpi=300)


#%%

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.loader import DataLoader
from src.utils import _set_font_size

sns.set_theme()
sns.set(font_scale=1.5)


data_loader = DataLoader()

N = 1000
BURN_IN = 10
M = data_loader.M - BURN_IN
# M = np.minimum(40, M)


def value_function(
    coalition: np.ndarray,
    group_noise: np.ndarray,
    lambda_pen: float,
    if_print: bool = False,
) -> Any:
    assert coalition.shape[0] == N
    history = {
        "lambda_spot": [],
        "lambda_reserve": [],
        "lambda_balancing": [],
        "power": [],
        "profit_coalition": [],
        "Reserve profit coalition": [],
        "profit_activation": [],
        "penalty": [],
    }

    rng = np.random.default_rng(24)

    # simulate over a horizon of M days
    for m in range(M + BURN_IN):

        # get N uniform integer between 0 and 23
        x = rng.integers(0, 24, N)
        # NOTE: x represent any assets during a day. Could also be a stochastic process...

        # set values in placeholder to 1 for each row corresponding to index in x
        power = np.zeros((N, 24))
        power[np.arange(N), x] = 1

        # power now encodes when, e.g., a ventilation system is ON and OFF during the day.

        # prices
        # lambda_reserve = np.ones(24) * 0.1
        # lambda_spot = np.ones(24) * 0.5
        # lambda_balancing = lambda_spot + np.maximum(0, rng.uniform(-1.5, 1.5, 24))
        lambda_reserve = data_loader._lambda_mfrr[m, :]
        lambda_spot = data_loader._lambda_spot[m, :]
        lambda_balancing = data_loader._lambda_rp[m, :]

        # activation happens when lambda_balancing > lambda_spot
        up_regulation = lambda_balancing > lambda_spot

        # append prices to history
        history["lambda_spot"].append(lambda_spot)
        history["lambda_reserve"].append(lambda_reserve)
        history["lambda_balancing"].append(lambda_balancing)
        history["power"].append(power)

        if m < BURN_IN:
            continue

        ### AGGREGATED ASSETS ###
        p_reserve_agg = history["power"][-2][coalition, :].sum(axis=0)

        # calculate required activation of portfolio
        should_up_regulate = p_reserve_agg * up_regulation

        # calculate actual up-regulation of portfolio (should match with actual power consumption)
        actual_up_regulation = should_up_regulate - np.maximum(
            0,
            should_up_regulate
            - power[coalition, :].sum(axis=0)
            + power[group_noise & coalition, :].sum(axis=0),
        )

        # calculate activation profit
        profit_activation = actual_up_regulation * (lambda_balancing - lambda_spot)

        # calculate penalty for not delivering the required up-regulation
        penalty = np.maximum(0, should_up_regulate - actual_up_regulation) * lambda_pen

        # calculate reserve profit
        profit_reserve = p_reserve_agg * lambda_reserve

        # total profit
        profit = profit_activation + profit_reserve - penalty

        history["profit_coalition"].append(profit.sum())
        history["Reserve profit coalition"].append(profit_reserve.sum())
        history["profit_activation"].append(profit_activation.sum())
        history["penalty"].append(penalty.sum())

    if if_print:
        ### print results ###
        print(f"Profit coalition: {np.sum(history['profit_coalition'])}")
        print(
            f"Reserve profit coalition: {np.sum(history['Reserve profit coalition'])}"
        )

    return sum(history["profit_coalition"])


experiments = {}
lambda_pen_max = 5  # maximum penalty [DKK/kWh]]

G = 5  # groups, i.e., flexible providers

assert G <= N

import itertools

# binomial coefficient
from scipy.special import comb

for lambda_pen in np.linspace(0, lambda_pen_max, 10):

    ##### SHAPLEY VALUES PER GROUP #####

    # assign each n in N to a group g in G
    rng = np.random.default_rng(10)
    groups = rng.integers(1, G + 1, N)

    assert groups.shape == (N,)

    # find all possible subcoalitions
    subcoalitions = []  # type:ignore
    for g in range(1, G + 1):
        subcoalitions.extend(
            np.array(list(itertools.combinations(np.arange(1, G + 1), g)))
        )

    assert len(subcoalitions) == 2**G - 1

    print(f"Number of subcoalitions: {len(subcoalitions)}")

    # set group 1 assets to be stubborn and reject up-regulation signals
    group_noise = groups == 1

    print(f"Noise injected for group 1: {any(group_noise)}")

    shapley_values = np.empty(G)
    # calculate shapley values for all groups (i.e., flexible providers)
    for g in range(1, G + 1):
        player_g = groups == g
        print(f"Group {g} has {player_g.sum()} assets")

        # find all subcoalitions that includes player g in S and corresponding coalition without g
        marginal_contributions_g = []
        for coalition in subcoalitions:
            if g in coalition:
                coalition_without_g = coalition[coalition != g]

                # we are now ready to compute marginal contribution of player g in this coalition
                coalition_assets = np.isin(groups, coalition)
                coalition_assets_without_g = np.isin(groups, coalition_without_g)
                v = value_function(coalition_assets, group_noise, lambda_pen)
                v_without_g = (
                    value_function(coalition_assets_without_g, group_noise, lambda_pen)
                    if coalition_without_g.size > 0
                    else 0
                )

                # TODO: cache binomial coefficients
                mc = 1 / comb(G - 1, len(coalition) - 1) * (v - v_without_g)
                marginal_contributions_g.append(mc)

        assert marginal_contributions_g

        # mean marginal contribution over all subcoalitions
        sv = np.sum(marginal_contributions_g) / G
        #  assert individual rationality, i.e., shap value >= groups own coalition profit
        assert sv >= value_function(player_g, group_noise, lambda_pen)

        # save shapley value
        shapley_values[g - 1] = sv

    assert np.isclose(
        shapley_values.sum(),
        value_function(np.ones(N, dtype=bool), group_noise, lambda_pen),
    ), "Sum of shapley values should equal value of grand coalition"

    print(f"Shapley values: {shapley_values}")
    print(f"Sum of Shapley values: {shapley_values.sum()}")
    print(
        f"Value of grand coalition: {value_function(np.ones(N, dtype=bool), group_noise, lambda_pen)}"
    )

    experiments[lambda_pen] = shapley_values

#%%
# Plot effect of penalty on shapley values for group 1 and average shapley values for other groups
f, ax = plt.subplots(1, 1, figsize=(8, 6))
xrange = list(experiments.keys())
group1_profits = [experiments[x][0] for x in xrange]

# calculate profit of coalition without group 1 to show individual rationality
group_profit_without_1 = [value_function(~group_noise, group_noise, l) for l in xrange]
group_profit_with_1 = [
    value_function(np.ones(N, dtype=bool), group_noise, l) for l in xrange
]
group_profit_with_1 = [experiments[x][1:].sum() for x in xrange]

# calculate profit of coalition = {group 1} only to show individual rationality for group 1
group1_profits_on_its_own = [
    value_function(group_noise, group_noise, l) for l in xrange
]

# assert all(
#     np.isclose(a, b)
#     for a, b in zip(group_profit_with_1, [experiments[x].sum() for x in xrange])
# )
# # assert group profit wo. 1 is smaller_equal to group profit with 1
# assert all(
#     a >= b + c
#     for a, b, c in zip(group_profit_with_1, group_profit_without_1, group1_profits)
# )

ax.plot(xrange, group1_profits, label=r"$\phi_{\{1\}}$ incl. group 2-6", marker="o")

ax.plot(
    xrange,
    group_profit_with_1,
    label=r"$\phi_{\mathcal{G}/\{1\}}$ incl. group 1",
    marker="o",
)
ax.plot(
    xrange,
    group_profit_without_1,
    label=r"$\phi_{\mathcal{G}/\{1\}}$ excl. group 1",
    marker="o",
)

ax.plot(
    xrange,
    group1_profits_on_its_own,
    label=r"$\phi_{\{1\}}$ excl. group 2-6",
    marker="o",
)

# plot horizontal line on 0
ax.plot(
    [xrange[0], xrange[-1]],
    [0, 0],
    linestyle="--",
    color="black",
    alpha=0.5,
)
ax.set_xlabel("Penalty [DKK/kWh]")
ax.set_ylabel("Payment [DKK]")
ax.legend()
# sort legends
handles, labels = ax.get_legend_handles_labels()
_handles = handles[:1] + handles[-1:] + handles[1:3]
_labels = labels[:1] + labels[-1:] + labels[1:3]
ax.legend(_handles, _labels, loc="best")

_set_font_size(ax, 16)
plt.tight_layout()
# save figure
plt.savefig("tex/figures/shapley_values.png", dpi=300)
