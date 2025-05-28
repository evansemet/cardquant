import math
from dataclasses import dataclass, field, fields
from collections import Counter
from typing import Annotated, get_args, get_origin, Any, Callable
import numpy as np
from enum import Enum
import pandas as pd
from rich.console import Console
from rich.table import Table



def validate_annotated_fields(instance: Any) -> None:
    for f_in in fields(instance):
        value = getattr(instance, f_in.name)
        origin = get_origin(f_in.type)
        if origin is Annotated:
            base_type, constraint = get_args(f_in.type)
            if not constraint(value):
                raise ValueError(
                    f"Field '{f_in.name}' = {value} does not satisfy {constraint.__name__}"
                )
            if not isinstance(value, base_type):
                raise ValueError(
                    f"Field '{f_in.name}' = {value} is not of type {base_type.__name__}"
                )

@dataclass
class OptionValuation:
    theo: Annotated[float, lambda x_val: x_val >= -1e-6]
    delta: Annotated[float, lambda x_val: abs(x_val) <= 1+1e-6 or np.isnan(x_val)]
    gamma: Annotated[float, lambda x_val: -1e-6 <= x_val <= 1+1e-6 or np.isnan(x_val)]
    theta: Annotated[float, lambda x_val: x_val <= 1e-6 or np.isnan(x_val)]
    charm: Annotated[float, lambda x_val: abs(x_val) <= 1+1e-6 or np.isnan(x_val)]
    color: Annotated[float, lambda x_val: abs(x_val) <= 1+1e-6 or np.isnan(x_val)]

    def __post_init__(self) -> None:
        validate_annotated_fields(self)

@dataclass
class Option:
    strike: Annotated[int, lambda x_val: x_val >= 0]
    call: OptionValuation
    put: OptionValuation
    CTE: Annotated[int, lambda x_val: x_val >= 0]
    
    def __post_init__(self) -> None:
        validate_annotated_fields(self)

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
    seen_cards: list[int] = field(default_factory=list)
    strike_list: list[int] = field(default_factory=lambda: list(range(50, 91, 10)))
    deck: list[int] = field(default_factory=lambda: list(range(1, 14)) * 4)
    with_replacement: bool = field(default=False)
    calculate_all_greeks: bool = field(default=True)

    options: dict[int, Option] = field(default_factory=dict, init=False)
    future: float = field(init=False)
    _deck_max_sum: int = field(init=False) 
    _original_deck_config: list[int] = field(init=False)
    _original_deck_unique_cards: frozenset[int] = field(init=False)

    def __post_init__(self) -> None:
        self._original_deck_config = list(self.deck)
        self._original_deck_unique_cards = frozenset(self.deck)
        self._validate_strikes()
        self._replacement()
        self._perform_initial_calculations()

    def __repr__(self) -> str:
        records = [] 

        if self.calculate_all_greeks:
            for strike_val, option_obj in self.options.items(): 
                records.append({
                    'Strike': strike_val,
                    'OptionType': 'CALL',
                    'Theo': f"{option_obj.call.theo:.4f}",
                    'Delta Δ': f"{option_obj.call.delta:.4f}",
                    'Gamma Γ': f"{option_obj.call.gamma:.4f}",
                    'Theta Θ': f"{option_obj.call.theta:.4f}",
                    'Charm ψ': f"{option_obj.call.charm:.4f}",
                    'Color χ': f"{option_obj.call.color:.4f}"
                })
                records.append({
                    'Strike': strike_val,
                    'OptionType': 'PUT',
                    'Theo': f"{option_obj.put.theo:.4f}", 
                    'Delta Δ':f"{option_obj.put.delta:.4f}",
                    'Gamma Γ': f"{option_obj.put.gamma:.4f}",
                    'Theta Θ': f"{option_obj.put.theta:.4f}",
                    'Charm ψ': f"{option_obj.put.charm:.4f}",
                    'Color χ': f"{option_obj.put.color:.4f}"
                })
            columns = ['Strike', 'OptionType', 'Theo', 'Delta Δ', 'Gamma Γ', 'Theta Θ', 'Charm ψ', 'Color χ']
        
        else:
            for strike_val, option_obj in self.options.items():
                records.append({
                    'Strike': strike_val,
                    'OptionType': 'CALL',
                    'Theo': f"{option_obj.call.theo:.4f}",
                    'Delta Δ': f"{option_obj.call.delta:.4f}"
                })
                records.append({
                    'Strike': strike_val,
                    'OptionType': 'PUT',
                    'Theo': f"{option_obj.put.theo:.4f}",
                    'Delta Δ': f"{option_obj.put.delta:.4f}"
                })
            columns = ['Strike', 'OptionType', 'Theo', 'Delta Δ']
        
        df = pd.DataFrame.from_records(data=records, columns=columns)
        table_title = f"Future Valuation: {self.future:.4f}" 
        table = Table(title=table_title)
        for column_name in columns: 
            table.add_column(column_name, justify="right")

        for _,row_data in df.iterrows(): 
            table.add_row(*[str(row_data[col_name]) for col_name in columns]) 

        Console().print(table)
        return "" 
            
    def _validate_strikes(self) -> None: 
        self.strike_list = sorted(list(set(self.strike_list)))
        
    def _replacement(self) -> None: 
        if self.with_replacement:
            if not self._original_deck_unique_cards: 
                 self.deck = list(frozenset(self._original_deck_config)) * (self.n + len(self.seen_cards) + 5)
            else:
                self.deck = list(self._original_deck_unique_cards) * (self.n + len(self.seen_cards) + 5)

    @staticmethod
    def _deck_max_sum_with_seen(n_val: int, known_cards: list[int], deck_list: list[int]) -> int: 
        temp_deck = Counter(deck_list)
        temp_deck.subtract(Counter(known_cards)) 
        if any(count < 0 for count in temp_deck.values()):
            raise ValueError("Invalid known_cards or deck state for _deck_max_sum_with_seen.")
        
        max_sum_val, needed = 0, n_val 
        for rank_val in sorted(temp_deck.keys(), reverse=True): 
            if temp_deck[rank_val] > 0 and needed > 0:
                take = min(temp_deck[rank_val], needed)
                max_sum_val += take * rank_val
                needed -= take
                if needed == 0:
                    break
        return max_sum_val

    def _compute_sum_distribution_partial(self, leftover_counts: dict[int, int], n_draw: int) -> list[int]: 
        if n_draw == 0:
            max_s_val_for_zero_draw = self._deck_max_sum_with_seen(0, [], list(Counter(leftover_counts).elements())) 
            dp_row = [0] * (max_s_val_for_zero_draw + 1)
            if max_s_val_for_zero_draw >= 0: 
                 dp_row[0] = 1 
            return dp_row
            
        if n_draw < 0:
            raise ValueError("Cannot draw a negative number of cards.")
        
        current_sum_of_cards_in_leftover = sum(v_val for k_key,v_val in leftover_counts.items()) 
        if n_draw > current_sum_of_cards_in_leftover :
             raise ValueError(f"Cannot draw {n_draw} cards, only {current_sum_of_cards_in_leftover} remain in leftover_counts for _compute_sum_distribution_partial.")

        max_s_val_for_this_draw = self._deck_max_sum_with_seen(n_draw, [], list(Counter(leftover_counts).elements()))

        dp = [[0]*(max_s_val_for_this_draw+1) for _ in range(n_draw+1)]
        dp[0][0] = 1

        processed_sum_limit = 0
        for r_val in sorted(leftover_counts.keys()): 
            count_r = leftover_counts[r_val]
            if count_r == 0:
                continue
            
            for k_val in range(n_draw, -1, -1): 
                for s_val_loop_inner in range(processed_sum_limit, -1, -1): 
                    if dp[k_val][s_val_loop_inner] == 0:
                        continue
                    for m_val in range(1, min(count_r, n_draw - k_val) + 1): 
                        new_k = k_val + m_val
                        new_s = s_val_loop_inner + r_val * m_val
                        if new_s <= max_s_val_for_this_draw:
                            dp[new_k][new_s] += dp[k_val][s_val_loop_inner] * math.comb(count_r, m_val)
            processed_sum_limit += r_val * count_r
            processed_sum_limit = min(processed_sum_limit, max_s_val_for_this_draw)
        return dp[n_draw]

    def _compute_sum_dist_true_replacement(self, unique_cards_pool: frozenset[int], num_draws: int) -> list[int]:
        if num_draws == 0:
            return [1] 
        if not unique_cards_pool:
             if num_draws == 0: return [1]
             return [0] 
        
        max_val_of_unique_card = 0
        if unique_cards_pool:
            max_val_of_unique_card = max(unique_cards_pool)
        else:
            return [0] * 1

        max_sum_for_dp_table = num_draws * max_val_of_unique_card
        
        dp = [0] * (max_sum_for_dp_table + 1)
        dp[0] = 1 

        for _ in range(num_draws): 
            new_dp = [0] * (max_sum_for_dp_table + 1)
            for s_sum in range(max_sum_for_dp_table + 1):
                if dp[s_sum] > 0:
                    for card_val in unique_cards_pool:
                        if s_sum + card_val <= max_sum_for_dp_table:
                            new_dp[s_sum + card_val] += dp[s_sum]
            dp = new_dp
        return dp

    def _option_theos(self, known_cards: list[int], n_total: int, strike_val_param: int) -> OptionValues: 
        k_len = len(known_cards) 
        sum_seen = sum(known_cards)
        n_draw = n_total - k_len
        
        if n_draw < 0:
            raise ValueError("n_total < len(known_cards). Invalid scenario.")
        
        if n_draw == 0:
            call_val = max(sum_seen - strike_val_param, 0)
            put_val = max(strike_val_param - sum_seen, 0)
            return OptionValues(float(call_val), float(put_val))

        ways_for_sum = []
        total_combos = 0.0

        if self.with_replacement:
            if not self._original_deck_unique_cards and n_draw > 0:
                return OptionValues(np.nan, np.nan) 
            
            ways_for_sum = self._compute_sum_dist_true_replacement(self._original_deck_unique_cards, n_draw)
            if self._original_deck_unique_cards: 
                total_combos = float(len(self._original_deck_unique_cards) ** n_draw)
            else: 
                total_combos = 0.0 if n_draw > 0 else 1.0
        else: 
            leftover_deck_counts = Counter(self.deck) 
            leftover_deck_counts.subtract(known_cards)
            
            remaining_deck_size = sum(leftover_deck_counts.values())
            if remaining_deck_size < n_draw:
                return OptionValues(np.nan,np.nan) 

            ways_for_sum = self._compute_sum_distribution_partial(leftover_deck_counts, n_draw)
            total_combos = float(math.comb(remaining_deck_size, n_draw))

        if total_combos == 0: 
             return OptionValues(np.nan, np.nan) if n_draw > 0 else OptionValues(float(max(sum_seen - strike_val_param, 0)), float(max(strike_val_param - sum_seen, 0)))

        call_val, put_val = 0.0, 0.0
        for s_draw_offset, ways in enumerate(ways_for_sum): 
            if ways == 0:
                continue
            p_s = ways / total_combos 
            final_sum = sum_seen + s_draw_offset 
            call_val += p_s * max(final_sum - strike_val_param, 0)
            put_val += p_s * max(strike_val_param - final_sum, 0)
        return OptionValues(call_val, put_val)

    def _option_delta(self, known_cards: list[int], n_total: int, strike_val_param: int, option_type: OptionType) -> float: 
        k_len = len(known_cards) 
        sum_seen = sum(known_cards)
        n_draw = n_total - k_len

        if n_draw < 0: return 0.0
        if n_draw == 0:
            final_sum_at_expiry = sum_seen
            if option_type == OptionType.CALL:
                return 1.0 if final_sum_at_expiry > strike_val_param else 0.0
            elif option_type == OptionType.PUT:
                return -1.0 if final_sum_at_expiry < strike_val_param else 0.0 
            return 0.0

        ways_for_sum = []
        total_combos = 0.0

        if self.with_replacement:
            if not self._original_deck_unique_cards and n_draw > 0: return 0.0
            ways_for_sum = self._compute_sum_dist_true_replacement(self._original_deck_unique_cards, n_draw)
            if self._original_deck_unique_cards:
                total_combos = float(len(self._original_deck_unique_cards) ** n_draw)
            else:
                total_combos = 0.0 if n_draw > 0 else 1.0
        else:
            leftover_deck_counts = Counter(self.deck) 
            leftover_deck_counts.subtract(known_cards)
            remaining_deck_size = sum(leftover_deck_counts.values())
            if remaining_deck_size < n_draw: return 0.0
            ways_for_sum = self._compute_sum_distribution_partial(leftover_deck_counts, n_draw)
            total_combos = float(math.comb(remaining_deck_size, n_draw))
        
        if total_combos == 0: return 0.0

        itm_events = 0 
        for s_draw_offset, ways in enumerate(ways_for_sum):
            if ways == 0:
                continue
            final_sum = sum_seen + s_draw_offset
            if option_type == OptionType.CALL:
                if final_sum > strike_val_param: 
                    itm_events += ways
            elif option_type == OptionType.PUT:
                if final_sum < strike_val_param: 
                    itm_events -= ways 
            else:
                raise ValueError(f"option_type must be either 'CALL' or 'PUT', got {option_type=}.")
        
        return itm_events / total_combos if total_combos > 0 else 0.0
    
    def _option_gamma(self, known_cards: list[int], n_total: int, strike_val_param: int, option_type: OptionType) -> float: 
        if (n_total - len(known_cards)) == 0 : return 0.0 
        left_delta = self._option_delta(known_cards, n_total, strike_val_param-1, option_type) 
        right_delta = self._option_delta(known_cards, n_total, strike_val_param+1, option_type) 
        return abs((right_delta - left_delta) / 2.0) 

    def _option_gammas(self, known_cards: list[int], n_total: int, strike_val_param: int) -> OptionValues: 
        return OptionValues(
            self._option_gamma(known_cards, n_total, strike_val_param, OptionType.CALL), 
            self._option_gamma(known_cards, n_total, strike_val_param, OptionType.PUT)
        )

    def _option_deltas(self, known_cards: list[int], n_total: int, strike_val_param: int) -> OptionValues: 
        return OptionValues(
            self._option_delta(known_cards, n_total, strike_val_param, OptionType.CALL), 
            self._option_delta(known_cards, n_total, strike_val_param, OptionType.PUT) 
        )
        
    def _mean_remaining(self, known_cards: list[int]) -> float: 
        leftover_deck_counts = Counter(self.deck) 
        leftover_deck_counts.subtract(known_cards) 
        
        remaining_cards_list = [rank_val for rank_val, count in leftover_deck_counts.items() for _ in range(count) if count > 0] 
        return np.mean(remaining_cards_list) if remaining_cards_list else 0.0 
        
    def _change_time_greek(self, current_greek_val: OptionValues, function_to_call: Callable, known_cards: list[int], n_total: int, strike_val_param: int, mean_rem_val: float | None = None) -> OptionValues: 
        num_cards_to_draw_next = n_total - len(known_cards) 
        if num_cards_to_draw_next <= 0 : return OptionValues(0.0, 0.0) 

        mean_rem_val = mean_rem_val if mean_rem_val is not None else self._mean_remaining(known_cards)
        
        if not self._original_deck_unique_cards: 
            return OptionValues(np.nan, np.nan)

        bot, top = math.floor(mean_rem_val), math.ceil(mean_rem_val)
        
        available_for_next_step_draw = []
        temp_deck_counts_for_next_step = Counter()

        if self.with_replacement:
            available_for_next_step_draw = list(self._original_deck_unique_cards)
            for card_val in available_for_next_step_draw:
                temp_deck_counts_for_next_step[card_val] = float('inf') 
        else:
            temp_deck_counts_for_next_step = Counter(self.deck) 
            temp_deck_counts_for_next_step.subtract(Counter(known_cards))
            available_for_next_step_draw = [r_val for r_val,c_val in temp_deck_counts_for_next_step.items() if c_val > 0]

        if not available_for_next_step_draw: 
             return OptionValues(np.nan, np.nan) 
        
        min_avail = min(available_for_next_step_draw)
        max_avail = max(available_for_next_step_draw)

        bot = max(min_avail, bot) 
        top = min(max_avail, top)   
        
        actual_bot = -1
        if temp_deck_counts_for_next_step.get(bot, 0) > 0: actual_bot = bot
        else:
            for val_check in range(bot - 1, min_avail - 1, -1): 
                if temp_deck_counts_for_next_step.get(val_check,0) > 0: actual_bot = val_check; break
            if actual_bot == -1: 
                 for val_check in range(bot, max_avail + 1): 
                    if temp_deck_counts_for_next_step.get(val_check,0) > 0: actual_bot = val_check; break
        if actual_bot == -1 : actual_bot = min_avail 

        actual_top = -1
        if temp_deck_counts_for_next_step.get(top, 0) > 0: actual_top = top
        else:
            for val_check in range(top + 1, max_avail + 1):
                if temp_deck_counts_for_next_step.get(val_check,0) > 0: actual_top = val_check; break
            if actual_top == -1: 
                for val_check in range(top -1, min_avail -1, -1): 
                    if temp_deck_counts_for_next_step.get(val_check,0) > 0: actual_top = val_check; break
        if actual_top == -1 : actual_top = max_avail

        bot, top = actual_bot, actual_top

        if bot > top : 
            closest_card = min(available_for_next_step_draw, key=lambda x_val: abs(x_val-mean_rem_val)) 
            bot = top = closest_card
        
        if bot == top:
            if temp_deck_counts_for_next_step.get(bot, 0) == 0:
                 return OptionValues(np.nan, np.nan)
            decimal = 0.5 
        else:
            decimal = (mean_rem_val - bot) / (top - bot) if (top - bot) != 0 else 0.5 
        
        next_val_bot, next_val_top = OptionValues(np.nan, np.nan), OptionValues(np.nan, np.nan)
        
        temp_known_cards_bot = known_cards + [bot]
        if temp_deck_counts_for_next_step.get(bot,0) > 0:
             next_val_bot = function_to_call(temp_known_cards_bot, n_total, strike_val_param)
        
        temp_known_cards_top = known_cards + [top]
        if temp_deck_counts_for_next_step.get(top,0) > 0:
            if bot == top:
                 next_val_top = next_val_bot
            else: 
                next_val_top = function_to_call(temp_known_cards_top, n_total, strike_val_param)
        
        if np.isnan(next_val_bot.call) or np.isnan(next_val_top.call) or \
           np.isnan(current_greek_val.call) or np.isnan(next_val_bot.put) or \
           np.isnan(next_val_top.put) or np.isnan(current_greek_val.put) : 
            return OptionValues(np.nan, np.nan)

        res_call = (1 - decimal) * next_val_bot.call + decimal * next_val_top.call - current_greek_val.call
        res_put = (1 - decimal) * next_val_bot.put + decimal * next_val_top.put - current_greek_val.put
        return OptionValues(res_call, res_put)
        
    def _option_thetas(self, theo_vals: OptionValues, known_cards: list[int], n_total: int, strike_val_param: int, mean_rem_val: float | None = None) -> OptionValues: 
        return self._change_time_greek(theo_vals, self._option_theos, known_cards, n_total, strike_val_param, mean_rem_val)
        
    def _option_charms(self, delta_vals: OptionValues, known_cards: list[int], n_total: int, strike_val_param: int, mean_rem_val: float | None = None) -> OptionValues: 
        return self._change_time_greek(delta_vals, self._option_deltas, known_cards, n_total, strike_val_param, mean_rem_val)
        
    def _option_colors(self, gamma_vals: OptionValues, known_cards: list[int], n_total: int, strike_val_param: int, mean_rem_val: float | None = None) -> OptionValues: 
        return self._change_time_greek(gamma_vals, self._option_gammas, known_cards, n_total, strike_val_param, mean_rem_val)

    def _calculate_future_value(self) -> None: 
        if self.with_replacement:
            mean_rem_val = np.mean(list(self._original_deck_unique_cards)) if self._original_deck_unique_cards else 0.0
        else:
            leftover_deck_counts = Counter(self.deck) 
            leftover_deck_counts.subtract(self.seen_cards)
            remaining_cards_list = [rank_val for rank_val, count in leftover_deck_counts.items() for _ in range(count) if count > 0] 
            mean_rem_val = np.mean(remaining_cards_list) if remaining_cards_list else 0.0 
        
        self.future = sum(self.seen_cards) + mean_rem_val * (self.n - len(self.seen_cards))
        
    def _perform_initial_calculations(self) -> None: 
        n_remaining_draws = self.n - len(self.seen_cards)
        
        if self.with_replacement:
            max_unique_card = max(self._original_deck_unique_cards) if self._original_deck_unique_cards else 0
            self._deck_max_sum = n_remaining_draws * max_unique_card
        else:
            current_remaining_deck_counts = Counter(self.deck) - Counter(self.seen_cards) 
            self._deck_max_sum = CardValuation._deck_max_sum_with_seen(n_remaining_draws, [], list(current_remaining_deck_counts.elements()))
        
        self._calculate_future_value()

        for strike_val_loop in self.strike_list: 
            theo_vals = self._option_theos(self.seen_cards, self.n, strike_val_loop) 
            delta_vals = self._option_deltas(self.seen_cards, self.n, strike_val_loop) 
            
            gamma_vals = OptionValues(np.nan, np.nan) 
            theta_vals = OptionValues(np.nan, np.nan) 
            charm_vals = OptionValues(np.nan, np.nan) 
            color_vals = OptionValues(np.nan, np.nan) 

            if self.calculate_all_greeks:
                gamma_vals = self._option_gammas(self.seen_cards, self.n, strike_val_loop)
                
                if len(self.seen_cards) < self.n:
                    mean_rem_val = self._mean_remaining(self.seen_cards) 
                    theta_vals = self._option_thetas(theo_vals, self.seen_cards, self.n, strike_val_loop, mean_rem_val)
                    charm_vals = self._option_charms(delta_vals, self.seen_cards, self.n, strike_val_loop, mean_rem_val)
                    color_vals = self._option_colors(gamma_vals, self.seen_cards, self.n, strike_val_loop, mean_rem_val)
                else: 
                    theta_vals = OptionValues(0.0, 0.0) 
                    charm_vals = OptionValues(0.0, 0.0)
                    color_vals = OptionValues(0.0, 0.0)
            
            self.options[strike_val_loop] = Option(
                strike_val_loop,
                OptionValuation(theo_vals.call, delta_vals.call, gamma_vals.call, theta_vals.call, charm_vals.call, color_vals.call),
                OptionValuation(theo_vals.put, delta_vals.put, gamma_vals.put, theta_vals.put, charm_vals.put, color_vals.put),
                self.n - len(self.seen_cards) 
            )

    def add_card(self, new_card: int) -> None:
        if len(self.seen_cards) >= self.n:
            raise ValueError(f"Cannot add card. Already seen {len(self.seen_cards)} out of {self.n} cards.")

        if not self.with_replacement:
            original_deck_counts_for_validation = Counter(self._original_deck_config)
            current_seen_counts = Counter(self.seen_cards)
            if current_seen_counts[new_card] + 1 > original_deck_counts_for_validation.get(new_card, 0):
                raise ValueError(f"Cannot add card {new_card}. Not enough instances of this card remaining in the original deck.")
        else: 
            if new_card not in self._original_deck_unique_cards:
                 raise ValueError(f"Card value {new_card} is not a valid card type for this deck.")
        
        self.seen_cards.append(new_card)
        if self.with_replacement: 
            self._replacement() 
        self._perform_initial_calculations()
        
