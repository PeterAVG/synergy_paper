import pandas as pd

pd.options.mode.chained_assignment = None


def find_rp_price(x: pd.Series) -> float:
    if x.flag == 1 and x.flag_down == 0:
        return x.BalancingPowerPriceUpDKK
    elif x.flag_down == 1 and x.flag == 0:
        return x.BalancingPowerPriceDownDKK
    elif x.flag_down == 0 and x.flag == 0:
        return x.SpotPriceDKK
    elif x.flag_down == 1 and x.flag == 1:
        if (x.SpotPriceDKK - x.BalancingPowerPriceDownDKK) > (
            x.BalancingPowerPriceUpDKK - x.SpotPriceDKK
        ):
            return x.BalancingPowerPriceDownDKK
        else:
            return x.BalancingPowerPriceUpDKK
    else:
        raise Exception


class DataLoader:
    def __init__(self, year: int = 2022) -> None:
        self.year = year

        df_scenarios = pd.read_csv(
            "data/scenarios_v2.csv",
            parse_dates=["HourUTC"],
        ).query(f"HourUTC.dt.year == {self.year}")

        # Remove dates where there are less than 23 hours in a day
        dates_to_substract = (  # noqa
            df_scenarios.groupby(df_scenarios.Date)
            .flag.count()
            .to_frame()
            .query("flag <= 23")
            .index.values.tolist()
        )
        df_scenarios = df_scenarios.query("Date != @dates_to_substract")
        df_scenarios["lambda_rp"] = df_scenarios.apply(
            lambda x: find_rp_price(x), axis=1
        ).values

        lambda_spot = df_scenarios.SpotPriceDKK.values.reshape(-1, 24)
        lambda_rp = df_scenarios.lambda_rp.values.reshape(-1, 24)
        lambda_mfrr = df_scenarios.mFRR_UpPriceDKK.values.reshape(-1, 24)

        up_regulation_event = (lambda_rp > lambda_spot).astype(int)

        assert lambda_rp.shape == lambda_spot.shape
        assert lambda_rp.shape == lambda_mfrr.shape
        assert lambda_rp.shape == up_regulation_event.shape

        self._up_regulation_event = up_regulation_event
        self._lambda_rp = lambda_rp / 1000  # kWh
        self._lambda_spot = lambda_spot / 1000
        self._lambda_mfrr = lambda_mfrr / 1000
        self.M = lambda_rp.shape[0]
