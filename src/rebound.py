#%%

import numpy as np

from src.loader import DataLoader

data_loader = DataLoader()

N = 1000
BURN_IN = 10
M = data_loader.M - BURN_IN
M = np.minimum(M, 40)

experiments = {}


def calculate_actual_up_regulation_and_rebound_agg(
    prev_rebound: float,
    p_base: np.ndarray,
    p_reserve: np.ndarray,
):
    actual_up_regulation = np.empty_like(p_base)
    rebound = np.zeros_like(p_base)
    actual_power = np.empty_like(p_base)
    # prev_rebound = np.zeros_like(p_base)
    for t in range(p_base.shape[0]):
        actual_power[t] = p_base[t] - p_reserve[t] + prev_rebound
        actual_power[t] = max(0, actual_power[t])
        # record how much we actually up-regulated or down-regulated
        actual_up_regulation[t] = max(0, p_base[t] - actual_power[t])
        rebound[t] = max(0, actual_power[t] - p_base[t])
        # how much we need to rebound the next hour
        prev_rebound = actual_up_regulation[t]

    return actual_up_regulation, rebound, actual_power, prev_rebound


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
        "rebound_cost_individual": [],
        "rebound_cost_aggregated": [],
        "penalty_individual": [],
        "penalty_aggregated": [],
    }

    rng = np.random.default_rng(24)
    lambda_pen = 0.1

    # placeholder for rebound from previous day which can potentially happen in the first hour
    prev_rebound = np.zeros((n, 24))
    prev_rebound_agg = 0.0

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

        # append prices and power to history
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
        # NOTE: when there is a rebound, we don't up-regulate as much as we should
        # Hence, we adjust for the (previous day's) rebound in actual_up_regulation
        # NOTE: we only need to adjust the previous day's rebound as each asset
        # only has one activation hour per day
        actual_up_regulation = np.maximum(0, actual_up_regulation - prev_rebound)

        # NOTE: this only works as long as each asset only has one activation hour per day
        # calculate required rebound (always subsequent to activations)
        rebound = np.roll(actual_up_regulation, 1, axis=1)
        rebound[:, 0] = 0  # set first hour to 0
        rebound = rebound + prev_rebound  # add rebound from previous day
        # save rebound for next day
        prev_rebound = np.roll(actual_up_regulation, 1, axis=1)
        prev_rebound[:, 1:] = 0  # set hours 2:24 to 0

        # calculate activation profit
        profit_activation = actual_up_regulation * (lambda_balancing - lambda_spot)

        # calculate penalty for not delivering the required up-regulation
        penalty = np.maximum(0, should_up_regulate - actual_up_regulation) * lambda_pen

        # calculate reserve profit
        profit_reserve = p_reserve * lambda_reserve

        # calculate rebound cost
        rebound_cost = rebound * lambda_balancing

        # total profit
        profit = profit_activation + profit_reserve - penalty - rebound_cost

        history["profit_individual"].append(profit.sum())
        history["Reserve profit individual"].append(profit_reserve.sum())
        history["rebound_cost_individual"].append(rebound_cost.sum())
        history["penalty_individual"].append(penalty.sum())

        #####################################################################

        ### AGGREGATED ASSETS ###
        p_reserve_agg = history["power"][-2].sum(axis=0)

        # calculate required activation of portfolio
        should_up_regulate = p_reserve_agg * up_regulation

        # calculate actual up-regulation of portfolio (should match with actual power consumption)
        # actual_up_regulation = should_up_regulate - np.maximum(
        #     0, should_up_regulate - power.sum(axis=0)
        # )

        # get actual up-regulation and rebound for aggregated portfolio
        (
            actual_up_regulation,
            rebound,
            _,
            prev_rebound_agg,
        ) = calculate_actual_up_regulation_and_rebound_agg(
            prev_rebound_agg, power.sum(axis=0), should_up_regulate
        )

        # TODO: verify that 1:1 rebound is correct

        # calculate activation profit
        profit_activation = actual_up_regulation * (lambda_balancing - lambda_spot)

        # calculate penalty for not delivering the required up-regulation
        penalty = np.maximum(0, should_up_regulate - actual_up_regulation) * lambda_pen

        # calculate reserve profit
        profit_reserve = p_reserve_agg * lambda_reserve

        # calculate rebound cost
        rebound_cost = rebound * lambda_balancing

        # total profit
        profit = profit_activation + profit_reserve - penalty - rebound_cost

        history["profit_aggregated"].append(profit.sum())
        history["Reserve profit aggregated"].append(profit_reserve.sum())
        history["rebound_cost_aggregated"].append(rebound_cost.sum())
        history["penalty_aggregated"].append(penalty.sum())

    print(f"Profit individual: {np.sum(history['profit_individual'])}")
    print(f"Profit aggregated: {np.sum(history['profit_aggregated'])}")
    print(f"Rebound cost individual: {np.sum(history['rebound_cost_individual'])}")
    print(f"Rebound cost aggregated: {np.sum(history['rebound_cost_aggregated'])}")
    print(f"Penalty individual: {np.sum(history['penalty_individual'])}")
    print(f"Penalty aggregated: {np.sum(history['penalty_aggregated'])}")
    print(f"Reserve profit individual: {np.sum(history['Reserve profit individual'])}")
    print(f"Reserve profit aggregated: {np.sum(history['Reserve profit aggregated'])}")

    assert np.isclose(
        np.sum(history["Reserve profit individual"]),
        np.sum(history["Reserve profit aggregated"]),
    )
    # assert np.isclose(
    #     np.sum(history["rebound_cost_aggregated"]),
    #     np.sum(history["rebound_cost_individual"]),
    # )

    experiments[n] = history


# Conclusion: of course there is a huge synergy effect because of the Law of Large Numbers and
# Central Limit Theorem. The aggregated system is much more predictable than the individual assets.

#%%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import _set_font_size

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

_set_font_size(ax, 16, 14)
plt.tight_layout()

plt.savefig("tex/figures/synergy_effect_rebound_1.png", bbox_inches="tight", dpi=300)


# plot of x vs profits for each experiment
f, ax = plt.subplots(1, 1, figsize=(10, 5))
xrange = list(experiments.keys())
rebound_cost_individual = [
    np.sum(experiments[n]["rebound_cost_individual"]) for n in xrange
]
rebound_cost_aggregated = [
    np.sum(experiments[n]["rebound_cost_aggregated"]) for n in xrange
]
profit_individual = [np.sum(experiments[n]["profit_individual"]) for n in xrange]
profit_aggregated = [np.sum(experiments[n]["profit_aggregated"]) for n in xrange]
penalty_individual = [np.sum(experiments[n]["penalty_individual"]) for n in xrange]
penalty_aggregated = [np.sum(experiments[n]["penalty_aggregated"]) for n in xrange]
ax.plot(
    xrange,
    rebound_cost_individual,
    label="Individual assets rebound cost",
)
ax.plot(xrange, rebound_cost_aggregated, label="Aggregated assets rebound cost")
ax.plot(
    xrange,
    profit_individual,
    label="Individual assets profit",
)
ax.plot(xrange, profit_aggregated, label="Aggregated assets profit")
ax.plot(
    xrange,
    penalty_individual,
    label="Individual assets penalty",
)
ax.plot(xrange, penalty_aggregated, label="Aggregated assets penalty")
ax.set_xlabel("# of assets")
ax.set_ylabel("DKK")
ax.legend()

_set_font_size(ax, 16, 14)
plt.tight_layout()

# save to tex/figures folder
plt.savefig("tex/figures/synergy_effect_rebound_2.png", bbox_inches="tight", dpi=300)

##### CONCLUSION: main points #####

# Assumption: each asset rebound IMMEDIATELY after activation, 1-to-1.
# Assumption: each asset is independent of each other.

# 1. The synergy effect is LESS than without rebound cost
# 2. Portfolio encounter MUCH higher rebound costs than sum of individual assets,
#    because the portfolio is also ACTIVATED (up-regulated) MUCH more.
#    (Recall that the rebound ONLY happens after foregone consumption)
# 3. Likewise, the activation profit is MUCH higher for the portfolio
# 4. And the penalty for non-delivery is LESS for the portfolio

##### NEXT STEPS #####

# Original research question: can aggregation reduce the rebound cost?
# Premise: aggreagation makes capacity PREDICTABLE.
# New premise: capacity is totally predictable for both individual assets and portfolio.
#   (Hence, no synergy effect in activation).
#   Will this reduce the rebound cost? No, everything behaves identically.
# Should we then consider uncertain rebounds? That only makes sense if we benefit from
# predicting the aggregated (total) rebound more accurately. One example of this could be
# capacity limitations...
#
#
# Heterogeneous assets: One big zinc furnace + many small ventilation systems.
# Scale as no. of flexible demands increases (approximate Shapley value).
