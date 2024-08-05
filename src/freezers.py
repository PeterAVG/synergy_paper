#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.loader import DataLoader

data_loader = DataLoader()

N = 100
BURN_IN = 10
M = data_loader.M - BURN_IN
M = np.minimum(M, 40)
REBOUND = False

experiments = {}


# def calculate_actual_up_regulation_and_rebound_agg(
#     prev_rebound: float,
#     p_base: np.ndarray,
#     p_reserve: np.ndarray,
# ):
#     actual_up_regulation = np.empty_like(p_base)
#     rebound = np.zeros_like(p_base)
#     actual_power = np.empty_like(p_base)
#     # prev_rebound = np.zeros_like(p_base)
#     for t in range(p_base.shape[0]):
#         actual_power[t] = p_base[t] - p_reserve[t] + prev_rebound
#         actual_power[t] = max(0, actual_power[t])
#         # record how much we actually up-regulated or down-regulated
#         actual_up_regulation[t] = max(0, p_base[t] - actual_power[t])
#         rebound[t] = max(0, actual_power[t] - p_base[t])
#         # how much we need to rebound the next hour
#         prev_rebound = actual_up_regulation[t]

#     return actual_up_regulation, rebound, actual_power, prev_rebound


def simulate_freezer_consumption(
    h: np.ndarray, k: np.ndarray, a: np.ndarray, jump: np.ndarray
) -> np.ndarray:
    # p = lambda h, k, a, jump: np.maximum(
    #     np.cos(2 * np.pi / 24 * h - k) * a + jump + 2, 0
    # )
    return np.maximum(np.cos(2 * np.pi / 24 * h - k) * a + jump + 2, 0)


def calculate_actuals(
    prev_rebound: np.ndarray,  # shape (n,)
    prev_baseline_power: np.ndarray,  # shape (n,)
    baseline_power: np.ndarray,
    actual_power: np.ndarray,
    actual_up_regulation: np.ndarray,
    required_up_regulation: np.ndarray,
    actual_rebound: np.ndarray,
    dof: np.ndarray,
    portfolio_level: bool = False,
) -> None:
    if portfolio_level:
        assert prev_rebound.shape == (1,)
        assert prev_baseline_power.shape == (1, 24)
        assert baseline_power.shape == (1, 24)
        assert actual_power.shape == (1, 24)
        assert actual_up_regulation.shape == (1, 24)
        assert required_up_regulation.shape == (1, 24)
        assert actual_rebound.shape == (1, 24)
    # for each asset...
    for a in range(actual_power.shape[0]):
        if REBOUND:
            rebound = prev_rebound[a]
            rebound_pct = rebound / prev_baseline_power[a, -1]
            rebound_pct = 0.0 if np.isnan(rebound_pct) else rebound_pct
        else:
            rebound = 0.0
            rebound_pct = 0.0
        # for each hour...
        for t in range(actual_power.shape[1]):
            # NOTE: an asset is only able to deliver up-regulation for 1 consecutive hour,
            # hence, it can't up-regulate if it's supposed to rebound, i.e., when rebound > 0
            # if portfolio_level:
            #     # a portfolio can split up-regulation and rebound across assets
            #     # portfolio should rebound proportionally to how much it up-regulated
            #     # portfolio_rebound = rebound_pct * baseline_power[a, t]
            #     # rebound_part = rebound_pct * baseline_power[a, t]
            #     # rebound_part = rebound
            #     # free_part = (1 - rebound_pct) * baseline_power[a, t]
            #     # free_part = max(baseline_power[a, t] - rebound, 0)
            #     # actual_power[a, t] = (
            #     #     baseline_power[a, t] - min(free_part, required_up_regulation[a, t] + rebound_part) + rebound_part
            #     # )
            #     actual_power[a, t] = max(
            #         0,
            #         baseline_power[a, t]
            #         - required_up_regulation[a, t]
            #         + sum(actual_up_regulation[a, :t])
            #         - sum(actual_rebound[a, :t]),
            #     )
            # else:
            #     # where a single asset obviously can't
            #     if rebound > 0:
            #         actual_power[a, t] = baseline_power[a, t] + rebound
            #     else:
            #         actual_power[a, t] = max(
            #             0, baseline_power[a, t] - required_up_regulation[a, t]
            #         )
            # NOTE: use cumalative sum of up-regulation and rebound to calculate actual power

            # NOTE: we multiply degrees of freedom with required up-regulation such that
            # the asset can't up-regulate if it's not allowed to
            if REBOUND:
                actual_power[a, t] = max(
                    0,
                    baseline_power[a, t]
                    - required_up_regulation[a, t] * dof[a, t]
                    + sum(actual_up_regulation[a, :t])
                    - sum(actual_rebound[a, :t]),
                )
            else:
                actual_power[a, t] = max(
                    0,
                    baseline_power[a, t] - required_up_regulation[a, t] * dof[a, t],
                )

            actual_up_regulation[a, t] = max(
                0, baseline_power[a, t] - actual_power[a, t]
            )
            actual_rebound[a, t] = max(0, actual_power[a, t] - baseline_power[a, t])

            # calculate rebound for next hour
            if REBOUND:
                # assume rebound happens immediately and 1:1 with up-regulation
                rebound = actual_up_regulation[a, t]
                rebound_pct = rebound / baseline_power[a, t]
                rebound_pct = 0.0 if np.isnan(rebound_pct) else rebound_pct
            else:
                rebound = 0.0
                rebound_pct = 0.0
            assert rebound >= 0


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
    lambda_pen = 0.1 * 1
    size = (n, 24)
    prev_baseline_power = np.empty(size)

    # placeholder for rebound from previous day which can potentially happen in the first hour
    prev_rebound = np.zeros(n)
    prev_rebound_agg = np.zeros(1)

    # # degrees of freedom for each asset. Randomly drawn.
    # degrees_of_freedom = rng.integers(0, 2, size=size)
    # degrees_of_freedom = np.ones(size)

    # simulate over a horizon of M days
    for m in range(M + BURN_IN):

        # get N uniform integer between 0 and 23
        # x = rng.integers(0, 24, n)
        # # NOTE: x represent any assets during a day. Could also be a stochastic process...

        h = np.tile(np.arange(0, 24, 1).reshape(-1, 24), (size[0], 1))
        k = rng.uniform(-12, 12, size=size)
        a = np.maximum(0.5, rng.normal(2, 1, size=size))
        jump = rng.uniform(-0.5, 0.5, size=size)
        assert h.shape == (size[0], 24) == k.shape == a.shape == jump.shape
        x = simulate_freezer_consumption(h, k, a, jump)
        assert x.shape == (size[0], 24)
        assert min(x.flatten()) >= 0
        # plot mean and std of power
        # f,ax = plt.subplots(1,1)
        # pd.DataFrame(power).mean(axis=0).plot(ax=ax)
        # pd.DataFrame(power).std(axis=0).plot(ax=ax)
        # pd.Series(power[0,:]).plot(ax=ax)

        # simulate n freezers with uncertain consumption
        # power = np.zeros((n, 24))
        # power[np.arange(n), x] = 1
        baseline_power = x.copy()

        # degrees of freedom for each asset. Randomly drawn.
        # 1 indicates no restriction, 0 indicates no up-regulation.
        degrees_of_freedom = rng.integers(0, 2, size=size)
        # degrees_of_freedom = np.ones(size)

        # power now encodes when a freezer is consuming power

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
        history["power"].append(baseline_power)

        if m < BURN_IN:
            prev_baseline_power = baseline_power
            continue

        # we now simulate the synergy effect of aggregating stochastic freezers
        # versus simulating them individually

        # p_reserve is the expected power of the aggregated system and individual asset
        # based on previous day
        # NOTE: this is a quite naive "forecaste" or baseline approach

        ### INDIVIDUAL ASSETS ###
        p_reserve = history["power"][-2]  # * degrees_of_freedom

        # activation happens when lambda_balancing > lambda_spot
        up_regulation = lambda_balancing > lambda_spot

        # calculate required activation of individual assets
        required_up_regulation = p_reserve * up_regulation

        # calculate required up-regulation of individual assets (should match with  power consumption)
        # required_up_regulation = should_up_regulate - np.maximum(
        #     0, should_up_regulate - baseline_power
        # )
        # actual power of the individual assets (and leftover rebound from previous day)
        actual_power = np.empty_like(required_up_regulation)
        actual_up_regulation = np.empty_like(required_up_regulation)
        actual_rebound = np.empty_like(required_up_regulation)
        # actual_power[:, 0] = prev_rebound

        # now run per asset actual power consumption given all the above
        calculate_actuals(
            prev_rebound,
            prev_baseline_power,
            baseline_power,
            actual_power,
            actual_up_regulation,
            required_up_regulation,
            actual_rebound,
            degrees_of_freedom,
        )

        if REBOUND:
            # assert actual up-regulation is equal to actual rebound
            assert (
                np.allclose(actual_up_regulation.sum(), actual_rebound.sum())
                or np.allclose(
                    actual_up_regulation.sum() - actual_up_regulation[:, -1].sum(),
                    actual_rebound.sum(),
                )
                or np.allclose(
                    actual_up_regulation.sum() + prev_rebound.sum(),
                    actual_rebound.sum(),
                )
            )

            # sanity checks: actual power should sum to baseline power
            assert (
                np.allclose(actual_power.sum(), baseline_power.sum())
                or np.allclose(
                    actual_power.sum() + actual_up_regulation[:, -1].sum(),
                    baseline_power.sum(),
                )
                or np.allclose(
                    actual_power.sum() - prev_rebound.sum(), baseline_power.sum()
                )
            )

        # calculate rebound for first hour next day
        # prev_rebound = np.maximum(0, baseline_power[:, -1] - actual_power[:, -1])
        prev_rebound = actual_up_regulation[:, -1]
        assert np.all(prev_rebound >= 0)

        # calculate activation profit
        profit_activation = actual_up_regulation * (lambda_balancing - lambda_spot)

        # calculate penalty for not delivering the required up-regulation
        penalty = (
            np.maximum(0, required_up_regulation - actual_up_regulation) * lambda_pen
        )

        # calculate reserve profit
        profit_reserve = p_reserve * lambda_reserve

        # calculate rebound cost
        rebound_cost = actual_rebound * lambda_balancing

        # total profit
        profit = profit_activation + profit_reserve - penalty - rebound_cost

        history["profit_individual"].append(profit.sum())
        history["Reserve profit individual"].append(profit_reserve.sum())
        history["rebound_cost_individual"].append(rebound_cost.sum())
        history["penalty_individual"].append(penalty.sum())

        #####################################################################

        ### AGGREGATED ASSETS ###
        p_reserve_agg = history["power"][-2].sum(
            axis=0
        )  # * degrees_of_freedom).sum(axis=0)

        # calculate required activation of portfolio
        required_up_regulation = p_reserve_agg * up_regulation
        # capable up-regulation when taking into acount the degrees of freedom
        capable_up_regulation = (history["power"][-2] * degrees_of_freedom).sum(
            axis=0
        ) * up_regulation
        # convert capable_up_regulation (dof) to percentage of required up-regulation
        degrees_of_freedom_agg = capable_up_regulation / required_up_regulation
        degrees_of_freedom_agg[np.isnan(degrees_of_freedom_agg)] = 0.0
        # assert all are non-negative
        assert np.all(degrees_of_freedom_agg >= 0)
        assert np.all(degrees_of_freedom_agg <= 1)

        # calculate actual up-regulation of portfolio (should match with actual power consumption)
        # actual_up_regulation = should_up_regulate - np.maximum(
        #     0, should_up_regulate - power.sum(axis=0)
        # )

        # get actual up-regulation and rebound for aggregated portfolio
        # (
        #     actual_up_regulation,
        #     rebound,
        #     _,
        #     prev_rebound_agg,
        # ) = calculate_actual_up_regulation_and_rebound_agg(
        #     prev_rebound_agg, power.sum(axis=0), should_up_regulate
        # )

        # calculate required up-regulation of individual assets (should match with  power consumption)
        # required_up_regulation = should_up_regulate - np.maximum(
        #     0, should_up_regulate - baseline_power.sum(axis=0)
        # )
        required_up_regulation = required_up_regulation.reshape(1, -1)
        # actual power of the individual assets (and leftover rebound from previous day)
        actual_power = np.empty_like(required_up_regulation)
        actual_up_regulation = np.empty_like(required_up_regulation)
        actual_rebound = np.empty_like(required_up_regulation)
        # actual_power[0, 0] = prev_rebound_agg

        # now run for portfolio given all the above
        calculate_actuals(
            prev_rebound_agg,
            prev_baseline_power.sum(axis=0).reshape(1, -1),
            baseline_power.sum(axis=0).reshape(1, -1),
            actual_power,
            actual_up_regulation,
            required_up_regulation,
            actual_rebound,
            degrees_of_freedom_agg.reshape(1, -1),
            portfolio_level=True,
        )

        if REBOUND:
            # assert actual up-regulation is equal to actual rebound
            try:
                assert (
                    np.allclose(actual_up_regulation.sum(), actual_rebound.sum())
                    or np.allclose(
                        actual_up_regulation.sum() - actual_up_regulation[:, -1].sum(),
                        actual_rebound.sum(),
                    )
                    or np.allclose(
                        actual_up_regulation.sum() + prev_rebound_agg.sum(),
                        actual_rebound.sum(),
                    )
                )
            except AssertionError:
                print("\n\n")
                print(f"baseline_power: \n{baseline_power.sum(axis=0)}\n")
                print(f"actual_power: \n{actual_power}\n")
                print(f"actual_up_regulation: \n{actual_up_regulation}\n")
                print(f"required_up_regulation: \n{required_up_regulation}\n")
                print(f"actual_rebound: \n{actual_rebound}\n")
                print(f"prev_rebound_agg: \n{prev_rebound_agg}\n")
                print(
                    f"actual_up_regulation-actual_rebound: \n{actual_up_regulation.sum() - actual_rebound.sum()}\n"
                )
                print(
                    f"actual_power-baseline_power: \n{actual_power.sum() - baseline_power.sum()}\n"
                )

            # sanity checks: actual power should sum to baseline power
            assert (
                np.allclose(actual_power.sum(), baseline_power.sum())
                or np.allclose(
                    actual_power.sum() + actual_up_regulation[:, -1].sum(),
                    baseline_power.sum(),
                )
                or np.allclose(
                    actual_power.sum() - prev_rebound_agg, baseline_power.sum()
                )
            )

        # calculate rebound for first hour next day
        prev_rebound_agg = actual_up_regulation[0, -1].reshape(-1)

        # calculate activation profit
        profit_activation = actual_up_regulation * (lambda_balancing - lambda_spot)

        # calculate penalty for not delivering the required up-regulation
        penalty = (
            np.maximum(0, required_up_regulation - actual_up_regulation) * lambda_pen
        )

        # calculate reserve profit
        profit_reserve = p_reserve_agg * lambda_reserve

        # calculate rebound cost
        rebound_cost = actual_rebound * lambda_balancing

        # total profit
        profit = profit_activation + profit_reserve - penalty - rebound_cost

        history["profit_aggregated"].append(profit.sum())
        history["Reserve profit aggregated"].append(profit_reserve.sum())
        history["rebound_cost_aggregated"].append(rebound_cost.sum())
        history["penalty_aggregated"].append(penalty.sum())

        prev_baseline_power = baseline_power

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

#%%


### print summery statistics ###
print("\n\n")
print(f"Profit aggregated: {np.sum(history['profit_aggregated'])}")
print(f"Profit individual: {np.sum(history['profit_individual'])}")
print(
    f"Profit difference: {np.sum(history['profit_aggregated']) - np.sum(history['profit_individual'])}"
)

print("Rebound cost aggregated: ", np.sum(history["rebound_cost_aggregated"]))
print("Rebound cost individual: ", np.sum(history["rebound_cost_individual"]))
print(
    "Rebound cost difference: ",
    np.sum(history["rebound_cost_aggregated"])
    - np.sum(history["rebound_cost_individual"]),
)

print("Penalty aggregated: ", np.sum(history["penalty_aggregated"]))
print("Penalty individual: ", np.sum(history["penalty_individual"]))
print(
    "Penalty difference: ",
    np.sum(history["penalty_aggregated"]) - np.sum(history["penalty_individual"]),
)


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

_set_font_size(ax, 16, 20)
plt.tight_layout()


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
ax.set_ylabel("")
ax.legend()

_set_font_size(ax, 16, 20)
plt.tight_layout()
plt.show()

# save to tex/figures folder
# plt.savefig("tex/figures/synergy_effect_rebound.png", bbox_inches="tight", dpi=300)

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
