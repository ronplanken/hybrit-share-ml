import random
import numpy as np
from copy import deepcopy
from enum import Enum

random.seed(1)

C_BUST = 22

class PlayerAction(Enum):
	HIT = 0
	STAND = 1

class DealerPlayStrategy(Enum):
	GREEDY = 0
	REACH17 = 1

class PlayerVictoryState(Enum):
    WON = 0
    LOST_BY_POINTS = 1
    LOST_BY_BUST = 2
    DRAW = 3

class GamePhase(Enum):
    SETUP = 0
    PLAYER_TURN = 1
    DEALER_TURN = 2
    END_OF_ROUND = 3

class DealerStrategyFactory:
    
    def __init__(self, game):
        self.dealer_strategies = {
            DealerPlayStrategy.GREEDY: DealerStrategyGreedy(game),
            DealerPlayStrategy.REACH17: DealerStrategyReach17(game)
        }
    
    def create_strategy(self, strategy):
        return self.dealer_strategies.get(strategy)
        

class DealerStrategy:
    
    def __init__(self, game):
        self.game = game
    
    def run(self, hand_total_to_beat):
        raise Exception()
    
    def hit(self):
        self.game.hit(self.game.dealer)

class DealerStrategyGreedy(DealerStrategy):
    
    def run(self, hand_total_to_beat):
        while hand_total_to_beat > self.game.dealer.hand.calculate_total():
            self.hit()

class DealerStrategyReach17(DealerStrategy):
    
    C_POINTS_GOAL = 17
    
    def run(self, hand_total_to_beat):
        dealer_total = self.game.dealer.hand.calculate_total()
        while dealer_total < hand_total_to_beat and dealer_total < self.C_POINTS_GOAL:
            self.hit()
            dealer_total = self.game.dealer.hand.calculate_total()

class PlayerState:
    
    def __init__(self, player_hand, dealer_revealed_card, remaining_cards):
        self.player_hand = player_hand
        self.dealer_revealed_card = dealer_revealed_card
        self.remaining_cards = remaining_cards
    
    def __str__(self):
        cards = ", ".join([x.__str__() for x in self.remaining_cards])
        return "Player {}, Dealer revealed card: {}\nRemaining cards: [{}]"\
            .format(self.player_hand.__str__(), self.dealer_revealed_card.__str__(), cards)

class RoundState:
    
    def __init__(self, has_round_ended, player_victory_state=None, player_total=None, dealer_total=None):
        self.has_round_ended = has_round_ended
        self.player_victory_state = player_victory_state
        self.player_total = player_total
        self.dealer_total = dealer_total
    
    def __str__(self):
        return "Has round ended: {}, Player victory state: {}, Player total: {}, Dealer total: {}"\
            .format(self.has_round_ended, self.player_victory_state, self.player_total, self.dealer_total)

class Card:
    
    def __init__(self, name, value, suit=None):
        self.name = name
        self.value = value
        self.suit = suit
    
    def is_usable_ace(self):
        return self.value == 11
    
    def use_ace(self):
        self.value = 1

    def __eq__(self, other):
        return self.name == other.name and self.suit == other.suit
    
    def __str__(self):
        return "{} of {} ({})".format(self.name, self.suit, self.value)    

class Hand:
    
    def __init__(self):
        self.cards = []
    
    def add_card(self, card):
        self.cards.append(card)
        
    def has_usable_ace(self):
        return self.get_usable_ace() is not None
    
    def use_ace(self):
        ace = self.get_usable_ace()
        if ace is not None:
            ace.use_ace()
    
    def get_usable_ace(self):
        for card in self.cards:
            if card.value == 11:
                return card
        return None
    
    def calculate_total(self):
        return sum(c.value for c in self.cards)
    
    def is_bust(self):
        return self.calculate_total() >= C_BUST
    
    def __str__(self):
        return "Hand: [{}]".format(", ".join([x.__str__() for x in self.cards]))
    
class Game:
    
    def __init__(self, dealer_strategy):
        self.action_map = {
            PlayerAction.HIT: self.hit,
            PlayerAction.STAND: self.stand
        }
        self.dealer_strategy = DealerStrategyFactory(self).create_strategy(dealer_strategy)
    
    def next_round(self):
        """
        Start a new round. Set the participating players and deal the initial cards.
        
        Returns
        -------
        PlayerState
            The game's state from the player's perspective.
        """
        self.game_phase = GamePhase.SETUP
        self.player = Player("Player")
        self.dealer = Player("Dealer")
        self.card_pool = deepcopy(INITIAL_CARD_POOL)
        self.remaining_cards = deepcopy(INITIAL_CARD_POOL)
        self.deal_initial_hands()

        for card in self.player.hand.cards:
            self.remaining_cards.remove(card)
        self.remaining_cards.remove(self.dealer.hand.cards[0])
        
        self.game_phase = GamePhase.PLAYER_TURN
        self.update_player_state()
        return self.state
    
    def update_player_state(self, drawn_card=None):
        if self.game_phase == GamePhase.PLAYER_TURN:
            if drawn_card is not None:
                self.remaining_cards.remove(drawn_card)
            self.state = PlayerState(self.player.hand, self.dealer.hand.cards[0], self.remaining_cards)
    
    def deal_initial_hands(self):
        for i in range(2):
            self.player.hand.add_card(self.deal_card())
            self.dealer.hand.add_card(self.deal_card())
        if self.player.hand.is_bust():
            self.player.hand.use_ace()
    
    def deal_card(self):
        r = random.choice(np.arange(len(self.card_pool)))
        card = self.card_pool[r]
        del self.card_pool[r]
        return card
    
    def hit(self, player=None):
        if not player:
            player = self.player

        drawn_card = self.deal_card()
        player.hand.add_card(drawn_card)
        self.update_player_state(drawn_card)

        if player.hand.is_bust():
            while player.hand.has_usable_ace() and player.hand.is_bust():
                player.hand.use_ace()
            
            self.update_player_state()
                
            if player.hand.is_bust():
                return self.get_round_end_state()
        return RoundState(False, None, self.player.hand.calculate_total(), self.dealer.hand.cards[0].value)
    
    def stand(self):
        self.game_phase = GamePhase.DEALER_TURN
        
        if self.dealer.hand.is_bust():
            self.dealer.hand.use_ace()

        self.dealer_strategy.run(self.player.hand.calculate_total())
        self.game_phase = GamePhase.END_OF_ROUND
        return self.get_round_end_state()
    
    def get_round_end_state(self):
        player_total = self.player.hand.calculate_total()
        dealer_total = self.dealer.hand.calculate_total()
        if self.player.hand.is_bust():
            return RoundState(True, PlayerVictoryState.LOST_BY_BUST, player_total, dealer_total)
        elif self.dealer.hand.is_bust():
            return RoundState(True, PlayerVictoryState.WON, player_total, dealer_total)
        elif player_total < dealer_total:
            return RoundState(True, PlayerVictoryState.LOST_BY_POINTS, player_total, dealer_total)
        elif player_total > dealer_total: 
            return RoundState(True, PlayerVictoryState.WON, player_total, dealer_total)
        else:
            return RoundState(True, PlayerVictoryState.DRAW, player_total, dealer_total)
    
    def act(self, action):
        """
        The actor performs the chosen action.
        
        Parameters
        ----------
        action : PlayerAction
            The action to perform.
		
        Returns
        -------
        [PlayerState, RoundState]
            The game's state from the player's perspective and the round state.
        """
        round_state = self.action_map.get(action)()
        return [self.state, round_state]
        

class Player:
    
    def __init__(self, name):
        self.name = name
        self.is_done = False
        self.hand = Hand()

card_types = [
    Card("Ace", 11),
    Card("King", 10),
    Card("Queen", 10),
    Card("Jack", 10),
    Card("10", 10),
    Card("9", 9),
    Card("8", 8),
    Card("7", 7),
    Card("6", 6),
    Card("5", 5),
    Card("4", 4),
    Card("3", 3),
    Card("2", 2)]
suits = ["spades", "hearts", "clubs", "diamonds"]
INITIAL_CARD_POOL = [Card(c.name, c.value, suit) for suit in suits for c in card_types]

ACTION_DESCRIPTION = {
    PlayerAction.HIT: "HIT",
    PlayerAction.STAND: "STAND"
}

def get_action_name(action):
    return ACTION_DESCRIPTION.get(action)