import math
from dataclasses import dataclass, field, fields
from itertools import product
from collections import Counter
from typing import Annotated, get_args, get_origin, Any
import numpy as np
from enum import Enum
import pandas as pd


def validate_annotated_fields(instance: Any, round_values: bool) -> None:
    for f in fields(instance):
        value = getattr(instance, f.name)
        origin = get_origin(f.type)
        if origin is Annotated:
            base_type, constraint = get_args(f.type)
            if not constraint(value):
                raise ValueError(
                    f"Field '{f.name}' = {value} does not satisfy {constraint.__name__}"
                )
            if not isinstance(value, base_type):
                raise ValueError(
                    f"Field '{f.name}' = {value} is not of type {base_type.__name__}"
                )

        if round_values and isinstance(value, float):
            rounded = round(value, 4)
            setattr(instance, f.name, rounded)


@dataclass
class OptionValuation:
    theo: Annotated[float, lambda x: x >= -1e-6]
    delta: Annotated[float, lambda x: abs(x) <= 1+1e-6]
    gamma: Annotated[float, lambda x: -1e-6 <= x <= 1+1e-6]
    theta: Annotated[float, lambda x: x <= 1e-6]

    def __post_init__(self) -> None:
        validate_annotated_fields(self, True)


@dataclass
class Option:
    strike: Annotated[int, lambda x: x >= 0]
    call: OptionValuation
    put: OptionValuation
    CTE: Annotated[int, lambda x: x >= 0]
    
    def __post_init__(self) -> None:
        validate_annotated_fields(self, False)


class OptionType(str, Enum):
    CALL = 'CALL'
    PUT = 'PUT'


@dataclass
class OptionValues:
    call: Any
    put: Any


@dataclass
class CardValuation:
    n: int = field(default=10)
    strike_list: list[int] = field(default_factory=lambda: list(range(50, 91, 10)))
    seen_cards: list[int] = field(default_factory=list)
    options: dict[int, Option] = field(default_factory=dict)
    deck: list[int] = field(default_factory=lambda: list(range(1, 14)) * 4)
    deck_max_sum: int = field(init=False)


    def __post_init__(self) -> None:
        self.validate_strikes()
        self.calculate_all_greeks_and_theos()


    def __repr__(self) -> str:
        records = list[dict[str, int | float | str]]()
        for strike, option in self.options.items():
            records.append({
                'Strike': strike,
                'OptionType': 'CALL',
                'Theo': option.call.theo,
                'Delta': option.call.delta,
                'Gamma': option.call.gamma,
                'Theta': option.call.theta
            })
            records.append({
                'Strike': strike,
                'OptionType': 'PUT',
                'Theo': option.put.theo,
                'Delta': option.put.delta,
                'Gamma': option.put.gamma,
                'Theta': option.put.theta
            })
        
        return pd.DataFrame.from_records(records, columns=['Strike', 'OptionType', 'Theo', 'Delta', 'Gamma', 'Theta']).to_string(index=False)


    def validate_strikes(self) -> None:
        self.strike_list = sorted(list(set(self.strike_list)))
        

    @staticmethod
    def deck_max_sum_with_seen(n: int, known_cards: list[int], deck: list[int]) -> int:
        leftover = Counter(deck)
        leftover.subtract(known_cards)
        if any(count < 0 for count in leftover.values()):
            raise ValueError("Invalid known_cards configuration. More copies than possible.")
        
        max_sum, needed = 0, n
        for rank in sorted(leftover.keys(), reverse=True):
            if leftover[rank] > 0:
                take = min(leftover[rank], needed)
                max_sum += take * rank
                needed -= take
                if needed == 0:
                    break
                
        return max_sum


    def compute_sum_distribution_partial(self, leftover: dict[int, int], n_draw: int) -> list[int]:
        if n_draw > sum(leftover.values()):
            raise ValueError("Cannot draw more cards than remain in leftover deck.")

        max_sum = self.deck_max_sum
        dp = [[0]*(max_sum+1) for _ in range(n_draw+1)]
        dp[0][0] = 1

        sorted_ranks = sorted(leftover.keys())
        for r in sorted_ranks:
            count_r = leftover[r]
            dp2 = [[0]*(max_sum+1) for _ in range(n_draw+1)]
            for k,s in product(range(n_draw+1), range(max_sum+1)):
                ways_before = dp[k][s]
                if ways_before == 0:
                    continue
                max_m = min(count_r, n_draw - k)
                for m in range(max_m+1):
                    new_k = k + m
                    new_s = s + r*m
                    if new_s <= max_sum:
                        dp2[new_k][new_s] += ways_before * math.comb(count_r, m)
            dp = dp2

        return dp[n_draw]


    def option_theos(self, known_cards: list[int], n_total: int, strike: int) -> OptionValues:
        k = len(known_cards)
        sum_seen = sum(known_cards)
        n_draw = n_total - k
        if n_draw < 0:
            raise ValueError("n_total < len(known_cards). Invalid scenario.")

        leftover = Counter(self.deck)
        leftover.subtract(known_cards)
        ways_for_sum = self.compute_sum_distribution_partial(leftover, n_draw)
        total_combos = math.comb(len(self.deck) - k, n_draw)

        call_val, put_val = 0.0, 0.0
        for s_draw, ways in enumerate(ways_for_sum):
            if ways == 0:
                continue
            p_s = ways / total_combos
            final_sum = sum_seen + s_draw
            call_val += p_s * max(final_sum - strike, 0)
            put_val += p_s * max(strike - final_sum, 0)
        return OptionValues(call_val, put_val)


    def option_delta(self, known_cards: list[int], n_total: int, strike: int, option_type: OptionType) -> float:
        k = len(known_cards)
        sum_seen = sum(known_cards)
        n_draw = n_total - k

        leftover = Counter(self.deck)
        leftover.subtract(known_cards)
        ways_for_sum = self.compute_sum_distribution_partial(leftover, n_draw)
        total_combos = math.comb(len(self.deck) - k, n_draw)

        count_condition = 0
        for s_draw, ways in enumerate(ways_for_sum):
            if ways == 0:
                continue
            final_sum = sum_seen + s_draw

            match option_type:
                case OptionType.CALL:
                    if final_sum > strike: 
                        count_condition += ways
                case OptionType.PUT:
                    if final_sum < strike: 
                        count_condition -= ways
                case _:
                    raise ValueError(f"option_type must be either 'CALL' or 'PUT', got {option_type=}.")

        return count_condition / total_combos
    
    
    def option_gamma(self, known_cards: list[int], n_total: int, strike: int, option_type: OptionType) -> float:
        left = self.option_delta(known_cards, n_total, strike-1, option_type)
        right = self.option_delta(known_cards, n_total, strike+1, option_type)
        return abs((right - left) / 2)


    def option_gammas(self, known_cards: list[int], n_total: int, strike: int) -> OptionValues:
        return OptionValues(
            self.option_gamma(known_cards, n_total, strike, OptionType.CALL), 
            self.option_gamma(known_cards, n_total, strike, OptionType.PUT)
        )


    def option_deltas(self, known_cards: list[int], n_total: int, strike: int) -> OptionValues:
        return OptionValues(
            self.option_delta(known_cards, n_total, strike, OptionType.CALL), 
            self.option_delta(known_cards, n_total, strike, OptionType.PUT)
        )

        
    def option_thetas(self, theo: OptionValues, known_cards: list[int], n_total: int, strike: int) -> OptionValues:
        leftover = Counter(self.deck)
        leftover.subtract(known_cards)
        remaining_cards = [rank for rank, count in leftover.items() for _ in range(count) if count > 0]
        mean_remaining = np.mean(remaining_cards) if remaining_cards else 0
        bot, top = math.floor(mean_remaining), math.ceil(mean_remaining)
        decimal = mean_remaining - bot
        self.deck_max_sum = CardValuation.deck_max_sum_with_seen(self.n, known_cards + [bot], self.deck)
        next_val_bot = self.option_theos(known_cards + [bot], n_total, strike)
        self.deck_max_sum = CardValuation.deck_max_sum_with_seen(self.n, known_cards + [top], self.deck)
        next_val_top = self.option_theos(known_cards + [top], n_total, strike)
        self.deck_max_sum = CardValuation.deck_max_sum_with_seen(self.n, known_cards, self.deck)
        return OptionValues(
            (1 - decimal) * next_val_bot.call + decimal * next_val_top.call - theo.call,
            (1 - decimal) * next_val_bot.put + decimal * next_val_top.put - theo.put
        ) 


    def calculate_all_greeks_and_theos(self) -> None:
        self.deck_max_sum = CardValuation.deck_max_sum_with_seen(self.n, self.seen_cards, self.deck)
        for strike in self.strike_list:        
            theo = self.option_theos(self.seen_cards, self.n, strike)
            delta = self.option_deltas(self.seen_cards, self.n, strike)
            gamma = self.option_gammas(self.seen_cards, self.n, strike)            
            theta = self.option_thetas(theo, self.seen_cards, self.n, strike)
            self.options[strike] = Option(
                strike,
                OptionValuation(theo.call, delta.call, gamma.call, theta.call if len(self.seen_cards) < self.n else 0),
                OptionValuation(theo.put, delta.put, gamma.put, theta.put if len(self.seen_cards) < self.n else 0),
                self.n - len(self.seen_cards)
            )
