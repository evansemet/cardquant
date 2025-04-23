from dataclasses import dataclass, field
from math import comb
from itertools import permutations
from typing import Annotated, get_type_hints, get_origin, get_args
from collections import Counter



@dataclass
class Probability:
    _12: Annotated[float, lambda x: x>=0.0] = field(default=0.0)
    _10: Annotated[float, lambda x: x>=0.0] = field(default=0.0)
    _8: Annotated[float, lambda x: x>=0.0] = field(default=0.0)


    def update_ways(self, n: int, ways: int) -> None:
        mapping = {12: '_12', 10: '_10', 8: '_8'}
        try:
            attr = mapping[n]
        except KeyError:
            raise ValueError(
                f"Invalid value of n. Got {n}. Must be one of {set(mapping.keys())}"
            )
        setattr(self, attr, getattr(self, attr) + ways)


    def convert_ways_to_prob(self, total_ways: int) -> None:
        self.__dict__ = {k: v/total_ways for k,v in self.__dict__.items()}



@dataclass
class PossibleDeck:
    diamonds: Annotated[int, lambda x: 0<=x<=12]
    hearts: Annotated[int, lambda x: 0<=x<=12]
    clubs: Annotated[int, lambda x: 0<=x<=12]
    spades: Annotated[int, lambda x: 0<=x<=12]
    
    probability: float = field(default=0)
    
    
    def __post_init__(self) -> None:
        self._validate_hand()
        
        
    def _validate_hand(self) -> None:
        if Counter([self.diamonds, self.hearts, self.clubs, self.spades]) != Counter([12,10,10,8]):
            raise ValueError("Not a valid possible deck")
             
                
    def __eq__(self, other: 'PossibleDeck') -> bool:
        return self.probability == other.probability
    
    
    def __lt__(self, other: 'PossibleDeck') -> bool:
        return self.probability < other.probability
    
    
    def __gt__(self, other: 'PossibleDeck') -> bool:
        return self.probability > other.probability
    
    
    def __gte__(self, other: 'PossibleDeck') -> bool:
        return self.probability >= other.probability
    
    
    def __lte__(self, other: 'PossibleDeck') -> bool:
        return self.probability <= other.probability



@dataclass
class Figgie:
    diamonds: Annotated[int, lambda x: 0<=x<=10] = field(default=0)
    hearts: Annotated[int, lambda x: 0<=x<=10] = field(default=0)
    clubs: Annotated[int, lambda x: 0<=x<=10] = field(default=0)
    spades: Annotated[int, lambda x: 0<=x<=10] = field(default=0)
    
    _p_diamonds: Probability = field(default_factory=lambda: Probability())
    _p_hearts: Probability = field(default_factory=lambda: Probability())
    _p_clubs: Probability = field(default_factory=lambda: Probability())
    _p_spades: Probability = field(default_factory=lambda: Probability())
    
    decks: list[PossibleDeck] = field(default_factory=list)
    
    
    def __post_init__(self) -> None:
        self._validate_hand()
        self._calculate_probabilities()
        
        
    def _validate_hand(self) -> None:
        for field_name,annotated_type in get_type_hints(self.__class__, include_extras=True).items():
            if get_origin(annotated_type) is Annotated:
                base_type, *validators = get_args(annotated_type)
                if base_type is int:
                    for validator in validators:
                        if callable(validator) and not validator((value := getattr(self, field_name))):
                            raise ValueError(
                                f"Validation failed for {field_name}: {value} does not satisfy {validator}"
                            )

        if (total := sum(filter(lambda n: isinstance(n, int), self.__dict__.values()))) not in (8, 10):
            raise ValueError(
                f"Hand size must be 8 or 10 for valid Figgie hand; got {total}."
            )
            
        
    @property
    def p_suit_objects(self) -> list[Probability]:
        return [
            self._p_diamonds,
            self._p_hearts,
            self._p_clubs,
            self._p_spades
        ]
        
        
    def _calculate_probabilities(self) -> None:
        total_ways = 0
        
        for (d,h,c,s) in set(permutations([12,10,10,8])):
            ways = (
                comb(d, self.diamonds) * 
                comb(h, self.hearts) * 
                comb(c, self.clubs) * 
                comb(s, self.spades)
            )
            total_ways += ways
            self.decks.append(PossibleDeck(d,h,c,s,ways))
        
            for suit_obj,value in zip(self.p_suit_objects, [d,h,c,s]):
                suit_obj.update_ways(value, ways)
        
        for deck in self.decks:
            deck.probability /= total_ways
            
        self.decks.sort(reverse=True)
        
        for suit_obj in self.p_suit_objects:
            suit_obj.convert_ways_to_prob(total_ways)
        
    
    def __repr__(self) -> str:

        suit_info = [
            ("Diamonds", "♦", self._p_hearts, "\033[91m"),
            ("Hearts",   "♥", self._p_diamonds,   "\033[91m"),
            ("Clubs",    "♣", self._p_spades,    "\033[30m"),
            ("Spades",   "♠", self._p_clubs,   "\033[30m"),
        ]
        
        sorted_suits = sorted(suit_info, key=lambda x: x[2]._12, reverse=True)
        
        header1 = f"\033[1m{'Suit':<10} | {'P(Goal Suit)':>12}\033[0m\n" + "─" * 25 + "\n"
        rows1 = "\n".join(
            f"{color}{symbol} {name:<8}\033[0m | {prob_obj._12 * 100:>10.2f}%"
            for name, symbol, prob_obj, color in sorted_suits
        )

        return header1 + rows1
        
