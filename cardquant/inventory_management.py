# from dataclasses import dataclass, field
# from card_memoization import OptionType, validate_annotated_fields, Option
# from typing import Annotated
# from itertools import product


# MAXIMUM_OPTION_POSITION = 10_000
# MINIMUM_OPTION_POSITION = -10_000
# STRIKE_LIST = list(range(50, 91, 10))


# @dataclass
# class Greeks:
#     delta: float = field(default=0)
#     gamma: float = field(default=0)
#     theta: float = field(default=0)


# @dataclass
# class OptionPosition:
#     strike: Annotated[int, lambda x: x in STRIKE_LIST]
#     option_type: Annotated[OptionType, lambda x: x in OptionType]
#     quantity: Annotated[int, lambda x: MINIMUM_OPTION_POSITION <= x <= MAXIMUM_OPTION_POSITION]

#     def __post_init__(self) -> None:
#         validate_annotated_fields(self, False)


# @dataclass
# class OptionHash:
#     strike: Annotated[int, lambda x: x in STRIKE_LIST]
#     option_type: Annotated[OptionType, lambda x: x in OptionType]

#     def __post_init__(self) -> None:
#         validate_annotated_fields(self, False)


# @dataclass
# class Inventory:
#     options: dict[OptionHash, OptionPosition] = field(default_factory=dict)
#     greeks: Greeks = field(default_factory=Greeks)
    
#     def __post_init__(self) -> None:
#         self.make_options()
        
#     def make_options(self) -> None:
#         for strike, option_type in product(STRIKE_LIST, OptionType):
#             self.options[OptionHash(strike, option_type)] = OptionPosition(strike, option_type, 0)

        
        