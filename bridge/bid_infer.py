from dataclasses import dataclass, field, InitVar
from importlib.metadata import distribution
from locale import currency
from typing import Literal, List

from bridge.constants import Suit, Side
from bridge.hand import Hand, Card
from bridge.bid import Bid
# from bridge.state import State, Result

import random
import numpy as np
from ipdb import set_trace as st
import copy
import math



def remain_to_hand(hands):
    for i in range(4):
        hands[i].remain_cards = hands[i].remain_cards + hands[i].cards_played
        hands[i].cards_played = []
    return hands

def bid_similarity(bid_gt, bid_generate):
    valid_bid_gt = [i for i in bid_gt if i != 0]
    valid_bid_generate = [i for i in bid_generate if i != 0]
    score = 0
    # exactly the same (same bid and position)
    for i, bid in enumerate(valid_bid_gt):
        if i < len(valid_bid_generate):
            if bid==valid_bid_generate[i] and bid.suit!=5:
                score+=10
    # gt appear in the simulated bidding
    for i, bid in enumerate(valid_bid_gt):
        if bid in valid_bid_generate and bid.suit!=5:
            score+=2
    return score


def Goren_bidding_infer_forward(state, unknown_sides):
    """
    bidding system inference (Goren), used during playing phase
    this funciton would gauss the hands using the bidding information
    """
    unknown_cards_list = [[] for i in range(4)]
    unknown_cards_list_played = [[] for i in range(4)]
    for side in unknown_sides:
        for card in state.hands[side].remain_cards:
            unknown_cards_list[card.suit].append(card) # remaining hands card: list of ['♠', '♥', '♦', '♣']
        for card in state.hands[side].cards_played:
            unknown_cards_list_played[card.suit].append(card) # played cards: list of ['♠', '♥', '♦', '♣'] of unknown side
    
    hcp_suit, count_suit = calculate_remaining_HCP(unknown_cards_list)
    _, played_count_suit = calculate_remaining_HCP(unknown_cards_list_played)
    original_count_suit = np.array(count_suit)+np.array(played_count_suit)
    original_count_suit = original_count_suit.tolist() # only for unknown sides
    
    # get bidding information
    dic_info = [None for i in range(4)]
    for i, current_bid in enumerate(state.bidding_info):
        if current_bid == 0:
            break
        who_bid = (state.declarer_starter + i) % 4
        current_bid = current_bid.to_str()
        if state.bidding_info[i-2] == 0:
            hcp_estimate, suit_estimate = first_bid_infer(current_bid)
        else:
            hcp_estimate, suit_estimate = response_bid_infer(current_bid)
        dic_info[who_bid] = [hcp_estimate, suit_estimate]
    # calculate player's rank
    rank_for_distribution, valid_rank = calculate_ranks(dic_info)
    
    # assign suit distribution according to the bidding info and player's rank
    assigned_suit_distribution = [[None] * 4 for i in range(4)]
    for i in rank_for_distribution:
        if i not in unknown_sides:
            continue
        if i not in valid_rank:
            continue
        suit_distribution = [None]*4
        for ii in range(4): # per suit
            this_suit = dic_info[i][1][ii]
            if this_suit!=None:
                # st()
                suit_distribution[ii] = np.random.choice(range(this_suit[0],this_suit[1]+1), p=calculate_prob(original_count_suit[ii],this_suit[0],this_suit[1],len(unknown_sides)))
        assigned_suit_distribution[i] = suit_distribution
    suit_gauss = [[None] * 4 for i in range(4)]
    for i in unknown_sides:
        suit_gauss[i] = fill_13(assigned_suit_distribution[i],13)
    # To give everyone 13 cards
    refined_assigned_suit_distribution = refine_suit_distribution(suit_gauss, unknown_sides, original_count_suit)

    # refined_assigned_suit_distribution is the original gauss (everyone has 13 cards), next, we minux the played cards.
    # st()
    played_suit = [[] for i in range(4)]
    for side in unknown_sides:
        player_suit_played = [0 for i in range(4)]
        for card in state.hands[side].cards_played:
            player_suit_played[card.suit]+=1 # played cards: list of ['♠', '♥', '♦', '♣'] of unknown side
        played_suit[side] = player_suit_played
    refined_assigned_suit_distribution_zero = copy.deepcopy(refined_assigned_suit_distribution)
    # print(refined_assigned_suit_distribution, played_suit)
    for side in unknown_sides:
        refined_assigned_suit_distribution_zero[side] = refined_assigned_suit_distribution[side] - np.array(played_suit[side])
    for side in range(4):
        if side in unknown_sides:
            continue
        else:
            refined_assigned_suit_distribution_zero[side] = np.array([0,0,0,0])
    refined_assigned_suit_distribution_non_zero = force_non_zero(refined_assigned_suit_distribution_zero, unknown_sides)
    # st()
    return refined_assigned_suit_distribution_non_zero.tolist()

def force_non_zero(mm, unknown_sides):
    refined_assigned_suit_distribution_zero = copy.deepcopy(mm)
    # print(mm)
    # if mm[2][0]==-1 and mm[2][3]==2:
        # st()
    for side in unknown_sides:
        assume_distri = refined_assigned_suit_distribution_zero[side]
        # st()
        for i in range(4):
            while assume_distri[i] < 0:
                refined_assigned_suit_distribution_zero[side][i]+=1
                suit_minus = np.random.choice([ii for ii in range(4) if assume_distri[ii]>0])
                # try:
                who_give = np.random.choice([ii for ii in unknown_sides if refined_assigned_suit_distribution_zero[ii][i]>0 and ii!=side])
                # except:
                #     st()
                refined_assigned_suit_distribution_zero[side][suit_minus]-=1
                refined_assigned_suit_distribution_zero[who_give][i]-=1
                refined_assigned_suit_distribution_zero[who_give][suit_minus]+=1
    # if -1 in assume_distri:
        # st()
    return refined_assigned_suit_distribution_zero


def Goren_bidding_infer(state, unknown_sides):
    """
    bidding system inference (Goren), used during playing phase
    this funciton would gauss the hands using the bidding information
    """
    unknown_cards_list = [[] for i in range(4)]
    for side in unknown_sides:
        for card in state.hands[side].remain_cards:
            unknown_cards_list[card.suit].append(card) # remaining hands card: list of ['♠', '♥', '♦', '♣']
    hcp_suit, count_suit = calculate_remaining_HCP(unknown_cards_list)
    count_suit_remaining= np.array(count_suit)
    num_cards = [len(state.hands[i].remain_cards) for i in range(4)]
    # get bidding information
    dic_info = [None for i in range(4)]
    for i, current_bid in enumerate(state.bidding_info):
        if current_bid == 0:
            break
        who_bid = (state.declarer_starter + i) % 4
        current_bid = current_bid.to_str()
        if state.bidding_info[i-2] == 0:
            hcp_estimate, suit_estimate = first_bid_infer(current_bid)
        else:
            hcp_estimate, suit_estimate = response_bid_infer(current_bid)
        dic_info[who_bid] = [hcp_estimate, suit_estimate]
    # calculate player's rank
    rank_for_distribution, valid_rank = calculate_ranks(dic_info)
    # assign suit distribution according to the bidding info and player's rank
    assigned_suit_distribution = [[None] * 4 for i in range(4)]
    for i in rank_for_distribution:
        if i not in unknown_sides:
            continue
        if i not in valid_rank:
            continue
        suit_distribution = [None]*4
        for ii in range(4): # per suit
            # np.random.choice(range(4,6), p=calculate_prob(12,4,5,3))
            # st()
            this_suit = dic_info[i][1][ii]
            if this_suit!=None:
                # st()
                suit_distribution[ii] = np.random.choice(range(this_suit[0],this_suit[1]+1), p=calculate_prob(count_suit[ii],this_suit[0],this_suit[1],len(unknown_sides)))
        assigned_suit_distribution[i] = suit_distribution
    # st()
    for i in valid_rank:
        suit_gauss = fill_13(assigned_suit_distribution[i],num_cards[i])
        if i in unknown_sides:
            count_suit_remaining-=np.array(suit_gauss)
        assigned_suit_distribution[i] = suit_gauss
    unvalid = [i for i in unknown_sides if i not in valid_rank]
    for i in range(len(unvalid)-1):
        suit_gauss = fill_13(assigned_suit_distribution[unvalid[i]], num_cards[unvalid[i]])
        assigned_suit_distribution[unvalid[i]] =suit_gauss
        count_suit_remaining-=np.array(suit_gauss)
    assigned_suit_distribution[unvalid[-1]] = count_suit_remaining.tolist()
    assigned_suit_distribution = refine_suit_distribution(assigned_suit_distribution, state, unknown_sides)
    return assigned_suit_distribution, unknown_cards_list

def refine_suit_distribution(suit_gauss, unknown_sides, original_count_suit):
    suit_gauss = np.array(suit_gauss)
    # print(f'{suit_gauss=}')
    # print(f'{unknown_sides=}')
    # print(f'{original_count_suit=}')
    for i in range(4):
        # print(suit_gauss)
        # st()
        while sum([suit_gauss[ii][i] for ii in unknown_sides])>original_count_suit[i]:
            who_decade = np.random.choice(unknown_sides)
            suit_gauss[who_decade][i] -= 1
            which_suit = np.random.choice(range(i+1,4))
            suit_gauss[who_decade][which_suit] += 1
        while sum([suit_gauss[ii][i] for ii in unknown_sides])<original_count_suit[i]:
            who_decade = np.random.choice(unknown_sides)
            suit_gauss[who_decade][i] += 1
            which_suit = np.random.choice(range(i+1,4))
            suit_gauss[who_decade][which_suit] -= 1
    return suit_gauss


def fill_13(suit_db, num_cards):
    suit = [i if i!=None else 0 for i in suit_db]
    to_fill = [i for i in range(4) if suit_db[i] == None]
    cnt = 0
    mn=len(to_fill) if len(to_fill)!=0 else 4
    # st()
    while sum(suit)<num_cards:
        if len(to_fill)!=0:
            index = to_fill[cnt%mn]
        else:
            index = 0
        suit[index] = suit[index] + 1
        cnt+=1
    while sum(suit)>num_cards:
        suit[-1] = suit[-1] - 1
    
    return suit

def fill_final_13(suit_db, num_cards):
    if suit_db[0] is None:
        return suit_db
    suit = copy.deepcopy(suit_db)
    while sum(suit)<num_cards:
        index = np.random.choice([0,1,2,3])
        suit[index] = suit[index] + 1
    while sum(suit)>num_cards:
        index = np.random.choice([0,1,2,3])
        suit[index] = suit[index] - 1
    return suit

def calculate_prob(total,low,high,num_players):
    sample = [i for i in range(low,high+1)]
    prob = []
    for i in sample:
        prob.append((math.comb(total, i) * (num_players-1)**(total-i) / num_players**total))
    try:
        prob = [ii/sum(prob) for ii in prob]
    except:
        prob = np.ones(len(sample))
        prob = [ii/sum(prob) for ii in prob]
    return prob

def calculate_ranks(dic_info):
    '''only suit, have not added hcp'''
    rank_players = []
    for i in range(len(dic_info)):
        suit_info = dic_info[i][1]
        distance=13
        for ss in suit_info:
            if ss != None:
                distance = min(ss[1]-ss[0], distance)
        rank_players.append(distance)
    v = sorted(range(len(rank_players)), key=lambda k: rank_players[k])
    valid = [i for i in range(4) if rank_players[i] < 13]
    return v, valid


def calculate_remaining_HCP(unknown_cards_list):
    '''
    input: 13 card
    return [SPADES: hcp,  HEARTS: hcp, DIAMONDS: hcp, CLUBS: hcp] , [len(x) for x in suits]
    Rule: 
        High-card points (HCP): A=4, K=3, Q=2, J=1
        Long-suit points: Add 1 point for a good 5-card suit, 2 for a 6-card suit, 3 for a 7-card suit.
        Short-suit points: If you have a trump fit with partner, add 1 point for a doubleton in a side suit, 2 for a singleton, 3 for a void.
    '''
    hcp_suit = [0, 0, 0, 0]
    count_suit = [0, 0, 0, 0]
    for hand_cards in unknown_cards_list:
        for hand_card in hand_cards:
            hcp_suit[hand_card.suit] += max(0, hand_card.rank-10)
            count_suit[hand_card.suit] += 1
    return hcp_suit, count_suit


def first_bid_infer(bid):
    # assert
    suit_estimate = [None for _ in range(4)] # ['♠', '♥', '♦', '♣']
    hcp_estimate = [0,40]
    balanced = [[1,13] for _ in range(4)]
    if bid == 'pass':
        hcp_estimate[1] = 4
    elif bid == '♣_1':
        hcp_estimate = [5,11]
        suit_estimate[2] = [3,3]
        suit_estimate[3] = [3,3]
    elif bid == '♦_1':
        hcp_estimate = [5,11]
        suit_estimate[2] = [4,4]
        suit_estimate[3] = [4,4]
    elif bid == '♥_1':
        hcp_estimate[0] = 12
        suit_estimate[1] = [5,6]
        suit_estimate[0] = [0,6]
    elif bid == '♠_1':
        hcp_estimate[0] = 12
        suit_estimate[0] = [5,6]
    elif bid == 'NT_1':
        hcp_estimate = [15,17]
        suit_estimate = copy.deepcopy(balanced)
    elif bid == '♣_2':
        hcp_estimate[0] = 22
    elif bid == '♦_2':
        hcp_estimate = [5,11]
        suit_estimate[2] = [7,13]
    elif bid == '♥_2':
        hcp_estimate = [5,11]
        suit_estimate[1] = [7,13]
    elif bid == '♠_2':
        hcp_estimate = [5,11]
        suit_estimate[0] = [7,13] # means spade>6
    elif bid == 'NT_2':
        hcp_estimate = [20,21]
        suit_estimate = copy.deepcopy(balanced)
    return hcp_estimate, suit_estimate

def response_bid_infer(bid):
    suit_estimate = [None for _ in range(4)] # ['♠', '♥', '♦', '♣']
    hcp_estimate = [0,40]
    balanced = [[1,13] for _ in range(4)]
    if bid == 'pass':
        pass
    elif bid == 'NT_2':
        hcp_estimate[0] = 13 
    elif bid == 'NT_3':
        hcp_estimate = [16,17]
    elif bid == '♠_1':
        hcp_estimate[0] = 6
        suit_estimate[0] = [4,13]
        suit_estimate[1] = [0,0]
    elif bid == 'NT_1':
        hcp_estimate = [6,9]
    elif bid == '♦_2':
        hcp_estimate[0] = 10
        suit_estimate[2] = [4,13]
    elif bid == '♣_2':
        hcp_estimate[0] = 10
        suit_estimate[3] = [4,13]
    elif bid == 'NT_2':
        hcp_estimate[0] = 13
    elif bid == '♥_3':
        hcp_estimate = [10,12]
        suit_estimate[1] = [3,13]
    elif bid == 'NT_3':
        hcp_estimate = [15,17]
        suit_estimate = copy.deepcopy(balanced)
        suit_estimate[1] = [3,13]
    elif bid == '♥_4':
        hcp_estimate[1] = 9
        suit_estimate[1] = [5,13]
    elif bid == '♠_2':
        hcp_estimate[0]=8
        suit_estimate[0] = [5,13]
    elif bid == '♥_2':
        hcp_estimate[0]=8
        suit_estimate[1] = [5,13]        
    # elif bid == '♥_2':
    #     hcp_estimate[0]=8
    #     suit_estimate[1] = [5,13]   
    return hcp_estimate, suit_estimate