# CardQuant 🃏 Quantitatively

CardQuant is a Python library designed for simulating and analyzing scenarios related to trading games. It currently features two main classes:

* **Figgie**: for the Jane Street Figgie game
* **CardValuation**: for IMC’s mock trading game involving options on the sum of drawn cards

---

## Features

* **Figgie Game Analysis**
  Calculate probabilities related to the Figgie card game.
* **Card Sum Options Valuation**
  Simulate a game where options are based on the sum of *n* drawn cards, and calculate their theoretical values and associated Greeks.

---

## Installation

```bash
pip install cardquant
```

---

## Core Classes

### Figgie

The `Figgie` class provides utilities for the Jane Street Figgie card game.

* **Primary function**: Calculate the probability of achieving the goal suit based on a given starting hand and game state.

### CardValuation

The `CardValuation` class simulates drawing *n* cards from a deck and pricing European‑style options on the final sum.

* **Outputs**:

  * **Theo**: theoretical option values
  * **Greeks**: Delta, Gamma, Theta, Charm, Color

---

## Key Concepts & Parameters

When creating a `CardValuation` instance, you can specify:

| Parameter                  | Description                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| **n**                      | Total number of cards to be drawn                                                              |
| **deck**                   | List representing the deck (e.g. `list(range(1,14))*4` for a standard deck)                    |
| **strike\_list**           | List of strike prices for which options will be valued                                         |
| **seen\_cards**            | List of cards already drawn and known                                                          |
| **with\_replacement**      | `True` to draw with replacement, `False` otherwise                                             |
| **calculate\_all\_greeks** | `True` to compute all Greeks (Theo, Delta, Gamma, Theta, Charm, Color); `False` for (Theo, Delta) |

After instantiation or any state change (e.g. via `add_card`), the instance exposes:

* **`options`**: dict mapping each strike to an object with Call/Put valuations (Theo + Greeks)
* **`future`**: expected final sum of the *n* cards given the current state

---

## Public Methods

* `add_card(new_card: int)`
  Adds `new_card` to `seen_cards` (validates availability and total count), then recalculates `future`, all `Theo`s, and Greeks.

---

## Greeks Explained

* **Delta (Δ)**
  Probability of the option expiring in the money.

  * Call Δ = P(final sum > K)
  * Put Δ  = −P(final sum < K)

* **Gamma (Γ)**
  Rate of change of Delta with respect to the strike:

  $$
    \Gamma \approx \frac{\bigl|\Delta(K+1) - \Delta(K-1)\bigr|}{2}
  $$

* **Time Greeks** (interpreted as drawing one more card):

  * **Theta (Θ)**: change in Theo if one more card (expected value) is drawn
  * **Charm (ψ)**: change in Delta under the same “one more card” move
  * **Color (χ)**: change in Gamma under the same “one more card” move

---

## Usage Example

```python
from cardquant import CardValuation

# initialize the pricing engine
# note that the values here are the default values used for IMC's mock trading
# nonetheless, the ability to alter them is given in the event slight changes are made to the mock
pricing_engine = CardValuation(
    n=10,
    seen_cards=[],
    strike_list=list(range(50,91,10)),
    deck=list(range(1,14))*4,
    with_replacement=False,
    calculate_all_greeks=True
)

print(pricing_engine)

# access the numerical value for the future
print(pricing_engine.future)

# access the numerical value for a particular attribute of an option strike
# by pricing_enging.options[strike].(call/put).(theo/greek)
print(pricing_engine.options[50].call.theo)
print(pricing_engine.options[50].put.delta)
print(pricing_engine.options[70].call.theta)

# Add a new card and recalc
pricing_engine.add_card(5)
print("\nAfter adding card 5:")
print(game)
```

---

## Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

Please adhere to the existing code style and include tests where appropriate.

---
