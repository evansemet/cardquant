import math
import random
from copy import deepcopy
from collections import Counter
import numpy as np
import sys
import os
from rich.console import Console
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from cardquant.IMC import CardValuation


def monte_carlo_single_strike_simulation(
    n_total_cards: int, 
    seen_cards_list: list[int], 
    original_deck_config: list[int], 
    strike: int, 
    n_simulations: int, 
    with_replacement_flag: bool
    ) -> tuple[float, float, float, float]:

    sum_of_seen_cards = sum(seen_cards_list)
    num_cards_to_draw = n_total_cards - len(seen_cards_list)

    total_call_payoff = 0.0
    total_put_payoff = 0.0
    call_itm_count = 0
    put_itm_count = 0

    if num_cards_to_draw < 0:
        raise ValueError("Monte Carlo: num_cards_to_draw cannot be negative.")
    
    if num_cards_to_draw == 0:
        final_sum = sum_of_seen_cards
        call_payoff = max(0, final_sum - strike)
        put_payoff = max(0, strike - final_sum)
        call_delta = 1.0 if final_sum > strike else 0.0
        put_delta = -1.0 if final_sum < strike else 0.0
        return call_payoff, put_payoff, call_delta, put_delta

    for _ in range(n_simulations):
        current_simulation_drawn_cards_sum = 0
        
        if with_replacement_flag:
            unique_cards_in_original_deck = list(frozenset(original_deck_config))
            if not unique_cards_in_original_deck:
                 raise ValueError("Monte Carlo: Original deck for replacement is empty.")
            drawn_this_sim = random.choices(unique_cards_in_original_deck, k=num_cards_to_draw)
            current_simulation_drawn_cards_sum = sum(drawn_this_sim)
        else:
            remaining_deck_for_sim = Counter(original_deck_config)
            remaining_deck_for_sim.subtract(Counter(seen_cards_list))
            
            available_cards_list = []
            for card_val, count in remaining_deck_for_sim.items():
                available_cards_list.extend([card_val] * count)

            if len(available_cards_list) < num_cards_to_draw:
                 current_simulation_drawn_cards_sum = sum(available_cards_list) 
            else:
                drawn_this_sim = random.sample(available_cards_list, num_cards_to_draw)
                current_simulation_drawn_cards_sum = sum(drawn_this_sim)

        final_sum = sum_of_seen_cards + current_simulation_drawn_cards_sum
        
        total_call_payoff += max(0, final_sum - strike)
        total_put_payoff += max(0, strike - final_sum)

        if final_sum > strike:
            call_itm_count += 1
        if final_sum < strike:
            put_itm_count += 1

    mc_call_theo = total_call_payoff / n_simulations
    mc_put_theo = total_put_payoff / n_simulations
    mc_call_delta = call_itm_count / n_simulations
    mc_put_delta = -1.0 * (put_itm_count / n_simulations)

    return mc_call_theo, mc_put_theo, mc_call_delta, mc_put_delta


def test_theos_and_deltas_with_monte_carlo(
    card_valuation_instance: CardValuation, 
    n_simulations: int, 
    theo_tolerance: float = 0.1, 
    delta_tolerance: float = 0.01 
    ):
    
    console = Console()
    console.print(f"\n[bold blue]Starting Monte Carlo Test (Simulations: {n_simulations})[/bold blue]")
    console.print(f"N: {card_valuation_instance.n}, Seen: {card_valuation_instance.seen_cards}, Replacement: {card_valuation_instance.with_replacement}, CalculateGreeks: {card_valuation_instance.calculate_all_greeks}")
    if hasattr(card_valuation_instance, '_original_deck_config'):
        console.print(f"Original Deck Config for MC: {card_valuation_instance._original_deck_config[:20]}{'...' if len(card_valuation_instance._original_deck_config) > 20 else ''}")
    else:
        console.print("[bold red]Warning: _original_deck_config attribute not found on instance.[/bold red]")


    results = []
    has_discrepancy = False

    for strike_val in card_valuation_instance.strike_list:
        console.print(f"\n--- Testing Strike: {strike_val} ---")
        
        analytical_option = card_valuation_instance.options.get(strike_val)
        if not analytical_option:
            console.print(f"[bold red]Error: Analytical option data not found for strike {strike_val}[/bold red]")
            results.append({"strike": strike_val, "status": "Error", "message": "Analytical data missing"})
            has_discrepancy = True
            continue

        ana_call_theo = analytical_option.call.theo
        ana_put_theo = analytical_option.put.theo
        ana_call_delta = analytical_option.call.delta
        ana_put_delta = analytical_option.put.delta

        try:
            mc_call_t, mc_put_t, mc_call_d, mc_put_d = monte_carlo_single_strike_simulation(
                n_total_cards=card_valuation_instance.n,
                seen_cards_list=deepcopy(card_valuation_instance.seen_cards),
                original_deck_config=card_valuation_instance._original_deck_config,
                strike=strike_val,
                n_simulations=n_simulations,
                with_replacement_flag=card_valuation_instance.with_replacement
            )
        except ValueError as e:
            console.print(f"[bold red]Monte Carlo Error for strike {strike_val}: {e}[/bold red]")
            results.append({"strike": strike_val, "status": "MC Error", "message": str(e)})
            has_discrepancy = True
            continue
        except AttributeError as e:
            console.print(f"[bold red]Attribute Error during MC setup (likely _original_deck_config missing from instance): {e}[/bold red]")
            results.append({"strike": strike_val, "status": "Attribute Error", "message": str(e)})
            has_discrepancy = True
            break 


        current_strike_results = {"strike": strike_val, "status": "OK", "details": []}
        strike_failed = False

        def check_value(name, ana_val, mc_val, tolerance):
            nonlocal strike_failed, has_discrepancy
            ana_is_nan = isinstance(ana_val, float) and np.isnan(ana_val)
            mc_is_nan = isinstance(mc_val, float) and math.isnan(mc_val)

            if ana_is_nan and mc_is_nan:
                is_fail = False 
            elif ana_is_nan or mc_is_nan:
                is_fail = True 
            else: 
                diff = abs(ana_val - mc_val)
                is_fail = diff > tolerance
            
            if is_fail:
                strike_failed = True
                has_discrepancy = True
            current_strike_results["details"].append(f"{name:10s}: Ana={ana_val:.4f}, MC={mc_val:.4f}, Diff={abs(ana_val - mc_val) if not (ana_is_nan or mc_is_nan) else 'NaN mismatch':.4f} {'[bold red](FAIL)[/bold red]' if is_fail else ''}")

        check_value("Call Theo", ana_call_theo, mc_call_t, theo_tolerance)
        check_value("Put Theo", ana_put_theo, mc_put_t, theo_tolerance)
        check_value("Call Delta", ana_call_delta, mc_call_d, delta_tolerance)
        check_value("Put Delta", ana_put_delta, mc_put_d, delta_tolerance)
        
        for detail in current_strike_results["details"]:
            console.print(detail)
        
        if strike_failed:
             current_strike_results["status"] = "FAIL"
             console.print(f"[bold yellow]Discrepancy found for strike {strike_val}[/bold yellow]")
        results.append(current_strike_results)

    console.print("\n[bold blue]Monte Carlo Test Summary:[/bold blue]")
    all_passed = not has_discrepancy
    for res in results:
        if res["status"] != "OK":
            console.print(f"Strike {res['strike']}: {res['status']} - {res.get('message', 'Comparison failed.')}")
    
    if all_passed:
        console.print("[bold green]All Theos and Deltas match Monte Carlo results within tolerance.[/bold green]")
    else:
        console.print("[bold red]Some Theos/Deltas FAILED to match Monte Carlo results.[/bold red]")
    
    return all_passed


if __name__ == '__main__':
    
    console_main = Console() 
    sim_count = 1_000_000
    theo_tol = 0.02
    delta_tol = 0.01 

    console_main.print("[bold magenta]== Test Case (Default Parameters) ==[/bold magenta]")
    try:
        cv_default = CardValuation()
        console_main.print(repr(cv_default))
        test_theos_and_deltas_with_monte_carlo(cv_default, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error during default parameters test: {e}[/bold red]")

    console_main.print("\n[bold magenta]== Test Case (Default Deck & Strikes, Custom N & Seen, No Replacement) ==[/bold magenta]")
    try:
        cv_dds_custom_n_seen_no_rep = CardValuation(
            n=5,
            seen_cards=[1, 7],
            with_replacement=False,
            calculate_all_greeks=False
        )
        console_main.print(repr(cv_dds_custom_n_seen_no_rep))
        test_theos_and_deltas_with_monte_carlo(cv_dds_custom_n_seen_no_rep, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error: {e}[/bold red]")

    console_main.print("\n[bold magenta]== Test Case (Default Deck & Strikes, Custom N & Seen, With Replacement) ==[/bold magenta]")
    try:
        cv_dds_custom_n_seen_rep = CardValuation(
            n=5,
            seen_cards=[1, 1, 7],
            with_replacement=True,
            calculate_all_greeks=False
        )
        console_main.print(repr(cv_dds_custom_n_seen_rep))
        test_theos_and_deltas_with_monte_carlo(cv_dds_custom_n_seen_rep, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error: {e}[/bold red]")
        
    console_main.print("\n[bold magenta]== Test Case (Default Deck & Strikes, calculate_all_greeks=False) ==[/bold magenta]")
    try:
        cv_dds_no_greeks = CardValuation(
            n=7,
            seen_cards=[10, 4],
            calculate_all_greeks=False
        )
        console_main.print(repr(cv_dds_no_greeks))
        test_theos_and_deltas_with_monte_carlo(cv_dds_no_greeks, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error: {e}[/bold red]")

    console_main.print("\n[bold magenta]== Test Case (Default Deck & Strikes, add_card) ==[/bold magenta]")
    try:
        cv_dds_add = CardValuation(
            n=6,
            calculate_all_greeks=False
        )
        console_main.print("Initial state (default deck & strikes):")
        console_main.print(repr(cv_dds_add))
        test_theos_and_deltas_with_monte_carlo(cv_dds_add, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
        
        cv_dds_add.add_card(13) 
        console_main.print("\nAfter adding card 13:")
        console_main.print(repr(cv_dds_add))
        test_theos_and_deltas_with_monte_carlo(cv_dds_add, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)

        cv_dds_add.add_card(2) 
        console_main.print("\nAfter adding card 2:")
        console_main.print(repr(cv_dds_add))
        test_theos_and_deltas_with_monte_carlo(cv_dds_add, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error: {e}[/bold red]")


    custom_params_base = {"n": 3, "strike_list": [10, 15, 20], "seen_cards": [5], "calculate_all_greeks": False, "deck": [1,2,3,4,5,6,7,8,9,10]*1}
    
    console_main.print("\n[bold magenta]== Initial Custom Test Case (No Replacement, Custom Deck/Strikes) ==[/bold magenta]")
    try:
        cv_test_no_rep_custom_deck = CardValuation(**custom_params_base, with_replacement=False)
        console_main.print(repr(cv_test_no_rep_custom_deck))
        test_theos_and_deltas_with_monte_carlo(cv_test_no_rep_custom_deck, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)

        cv_test_no_rep_custom_deck.add_card(10)
        console_main.print("\n[bold magenta]== After adding card 10 (No Replacement, Custom Deck/Strikes) ==[/bold magenta]")
        console_main.print(repr(cv_test_no_rep_custom_deck))
        test_theos_and_deltas_with_monte_carlo(cv_test_no_rep_custom_deck, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error during no replacement custom deck test: {e}[/bold red]")


    console_main.print("\n[bold magenta]== Custom Test Case (With Replacement, Custom Deck/Strikes) ==[/bold magenta]")
    try:
        cv_test_rep_custom_deck = CardValuation(**custom_params_base, with_replacement=True)
        console_main.print(repr(cv_test_rep_custom_deck))
        test_theos_and_deltas_with_monte_carlo(cv_test_rep_custom_deck, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
        
        cv_test_rep_custom_deck.add_card(1) 
        console_main.print("\n[bold magenta]== After adding card 1 (With Replacement, Custom Deck/Strikes) ==[/bold magenta]")
        console_main.print(repr(cv_test_rep_custom_deck))
        test_theos_and_deltas_with_monte_carlo(cv_test_rep_custom_deck, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error during with replacement custom deck test: {e}[/bold red]")


    console_main.print("\n[bold magenta]== Custom Test Case (All Cards Seen, Custom Deck/Strikes) ==[/bold magenta]")
    try:
        cv_all_seen_custom_deck = CardValuation(n=3, strike_list=[15, 20], seen_cards=[5,10,1], calculate_all_greeks=False, deck=[1,2,3,4,5,6,7,8,9,10])
        console_main.print(repr(cv_all_seen_custom_deck))
        test_theos_and_deltas_with_monte_carlo(cv_all_seen_custom_deck, n_simulations=1000, theo_tolerance=theo_tol, delta_tolerance=delta_tol) 
    except Exception as e:
        console_main.print(f"[bold red]Error during all cards seen custom deck test: {e}[/bold red]")

    console_main.print("\n[bold magenta]== Custom Test Case (Calculate_all_greeks = False, Custom Deck/Strikes) ==[/bold magenta]")
    try:
        cv_no_greeks_custom_deck = CardValuation(n=3, strike_list=[10,15], seen_cards=[5], calculate_all_greeks=False, deck=[1,2,3,4,5,6,7,8,9,10])
        console_main.print(repr(cv_no_greeks_custom_deck)) 
        test_theos_and_deltas_with_monte_carlo(cv_no_greeks_custom_deck, n_simulations=sim_count, theo_tolerance=theo_tol, delta_tolerance=delta_tol)
    except Exception as e:
        console_main.print(f"[bold red]Error during calculate_all_greeks=False custom deck test: {e}[/bold red]")

