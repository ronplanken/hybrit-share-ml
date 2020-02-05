import numpy as np
import pandas as pd
import random
from blackjack import PlayerVictoryState, PlayerAction, get_action_name
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

ACTIONS = [PlayerAction.HIT, PlayerAction.STAND]

def print_q_table(Q_table):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df = pd.DataFrame(columns=["HIT", "STAND"], data=Q_table.values(), index=Q_table.keys())
        df.sort_index(inplace=True)
        display(df)

def plot_victory_rates(victory_rates, nr_of_rounds):
    plt.figure(figsize=(15, 9))
    plt.xlabel("episode")
    plt.ylabel("ratio (%)")
    plt.plot(np.arange(nr_of_rounds), np.array(victory_rates)[:,0], label="wins", color='g')
    plt.plot(np.arange(nr_of_rounds), np.array(victory_rates)[:,1], label="losses", color='r')
    plt.plot(np.arange(nr_of_rounds), np.array(victory_rates)[:,2], label="draws", color='k')
    plt.plot(np.arange(nr_of_rounds), np.array(victory_rates)[:,0] + np.array(victory_rates)[:,2], label="wins & draws", color='c')
    plt.legend()
    plt.show()

def plot_rewards(rewards, factor):
    rr = []
    for i in range(int(len(rewards) / factor)):
        rr.append(sum(rewards[i * factor : (i+1) * factor]) / factor)
    plt.figure(figsize=(15, 9))
    plt.xlabel(f"episode ×{factor}")
    plt.ylabel("average reward")
    plt.scatter(np.arange(len(rewards) / factor), rr)
    plt.show()

def choose_action(Q_table, state, epsilon):
    """
    Pick a random action (to explore options) or pick the action with the highest weight from the Q-Table (to exploit prior
    knowledge), depending on the epsilon value.
    
    Parameters
    ----------
    Q_table : dict
        The Q-Table containing the action-state weights.
    
    state : str
        A textual representation of the game state.
    
    epsilon : float
        A number between 0-1. It decides whether to use the Q-Table weights to determine the next action (exploitating), or to
        take a random action (exploration). 0 = always use the Q-Table weights, 1 = always take a random action.
        

    Returns
    -------
    PlayerAction
        The player action to perform.
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    else:
        actions = Q_table.get(state.__str__())
        if actions is None:
            return random.choice(ACTIONS)
        else:
            return PlayerAction(np.argmax(actions))

def test_algorithm(game, Q_table, State, show_steps=True):
    lowest_win_draw_ratio = 100
    highest_win_draw_ratio = 0
    avg_victory_rates = []
    nr_of_iterations = 10
    nr_of_rounds = 1000
    
    for iteration in range(nr_of_iterations):
        print()
        print("==================================================")
        print(f"ITERATION {iteration + 1}")
        print("==================================================")
        wins = 0
        losses = 0
        draws = 0
        victory_rates = []

        for episode in range(nr_of_rounds):
            if show_steps:
                print()
                print("=========================")
                print("NEW ROUND")
                print("=========================")
            current_state_ = game.next_round()
            current_state = State(current_state_)
            done = False

            while not done:
                action = choose_action(Q_table, current_state, 0)
                next_state_, round_state = game.act(action)
                next_state = State(next_state_)

                if show_steps:
                    print()
                    print(f"state: {current_state}")
                    print(f"action: {get_action_name(action)}")
                    print(f"result: {round_state}")


                done = round_state.has_round_ended
                current_state = next_state

                if done:
                    player_victory_state = round_state.player_victory_state
                    if player_victory_state == PlayerVictoryState.WON:
                        wins += 1
                    elif player_victory_state == PlayerVictoryState.DRAW:
                        draws += 1
                    else:
                        losses += 1

                    victory_rates.append([
                        wins / (episode + 1) * 100,
                        losses / (episode + 1) * 100,
                        draws / (episode + 1) * 100
                    ])

        win_ratio = wins / nr_of_rounds * 100
        loss_ratio = losses / nr_of_rounds * 100
        draw_ratio = draws / nr_of_rounds * 100
        win_draw_ratio = win_ratio + draw_ratio
        avg_victory_rates.append([
            win_ratio,
            loss_ratio,
            draw_ratio,
            win_draw_ratio
        ])
        
        lowest_win_draw_ratio = min(lowest_win_draw_ratio, win_draw_ratio)
        highest_win_draw_ratio = max(highest_win_draw_ratio, win_draw_ratio)
        
        print()
        print(f"Wins: {win_ratio}% - Losses: {loss_ratio}% - Draws: {draw_ratio}%")
    plot_victory_rates(avg_victory_rates, nr_of_iterations)
    print()
    print(f"FINAL WIN DRAW RATIO: {lowest_win_draw_ratio}% - {highest_win_draw_ratio}%")