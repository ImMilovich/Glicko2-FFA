#!/usr/bin/env python
# coding: utf-8

# ### Running simulations back to back

# In[ ]:


import pickle
import random as ran
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlin
import math
import time
import copy
import trueskill as ts
import pandas as pd

from glicko2 import *
from scipy.stats import norm, skewnorm, lognorm, spearmanr, wilcoxon, pearsonr, skew, pareto, ttest_rel
from scipy.stats.stats import pearsonr
from scipy.integrate import quad
from sklearn.metrics import ndcg_score
from ipywidgets import FloatProgress
from IPython.display import display


# In[ ]:


sim_amount = 25  #how many different sims should be run - sets amount of columns in output
player_amount = 250 #how many players are in the sim
match_amount = 50 #how many matches each player should have played
gamesUntilPrint = 1 #every x match played it logs the statistics, one iteration of the while loop = 1 match played


# In[ ]:


def drawPointsForWinner(k):
    # points and weights have been modified to protect our industry partner
    pointList = [18, 19, 20]
    weights = [0.4, 7.6, 92]
    return ran.choices(pointList,weights=weights,k=k)


# In[ ]:


def drawPointsForSecond(k):
    # points and weights have been modified to protect our industry partner
    pointList = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    weights = [0.15, 0.12, 0.1, 0.085, 0.08, 0.07, 0.065, 0.06, 0.05, 
               0.04, 0.035, 0.03, 0.03, 0.025, 0.025, 0.02, 0.015]
    return ran.choices(pointList,weights=weights,k=k)


# In[ ]:


def drawPointsForThird(k, cutoff): #make sure that you can only draw at or below the value from the previous point draw
    
    # points and weights have been modified to protect our industry partner
    pointList = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    weights = [0.08, 0.07, 0.065, 0.12, 0.15, 0.1, 0.085, 0.06, 0.05, 
               0.04, 0.035, 0.03, 0.03, 0.025, 0.02, 0.02, 0.01, 0.01]
    
    cutoffIndex = pointList.index(cutoff)
    pointList = pointList[cutoffIndex:-1]
    weights = weights[cutoffIndex:-1]
    
    sumWeights = sum(weights)
    for i in range(0, len(weights)):
        weights[i] = weights[i]/sumWeights*100
    
    return ran.choices(pointList,weights=weights,k=k)


# In[ ]:


def drawPointsForFourth(k, cutoff): #make sure that you can only draw at or below the value from the previous point draw
    
    # points and weights have been modified to protect our industry partner
    pointList = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    weights = [0.08, 0.07, 0.065, 0.06, 0.05, 0.04, 0.15, 0.12, 0.1, 
               0.085, 0.035, 0.03, 0.03, 0.025, 0.02, 0.02, 0.01, 0.01]
    
    cutoffIndex = pointList.index(cutoff)
    pointList = pointList[cutoffIndex:-1]
    weights = weights[cutoffIndex:-1]
    
    sumWeights = sum(weights)
    for i in range(0, len(weights)):
        weights[i] = weights[i]/sumWeights*100
    
    return ran.choices(pointList,weights=weights,k=k)


# In[ ]:


#New method solving integrals (it's used inside the for loop later to calculate for each player)
def func_e(x, player):
    return math.exp((-(x-player.mu)**2)/(2*player.phi**2))

def func_prod(x, player):
    return norm.cdf((x-player.mu)/player.phi)

def func3(x, players):
    funcendelig = func_e(x, players[0])
    for i in range(1, len(players)):
        funcendelig = funcendelig*func_prod(x, players[i])
    return funcendelig

#Need to have this shift code and combine the three equations above in order for it to work in Python
def shift_integrand(integrand, offset):
    def dec(x, players):
        return integrand(x - offset, players)
    return dec

def my_quad(func, a, b, midpoint=0.0, **kwargs):
    if midpoint != 0.0:
        func = shift_integrand(func, -midpoint)
    return quad(func, a, b, **kwargs)


# In[ ]:


#Adjusting the skill ratings by normalizing them. If not we cannot solve the integrals in Python.
def adjust_rating(list_of_players):
    
    max_val = -5000000
    
    for i in range(0, len(list_of_players)):
        if list_of_players[i].mu > max_val:
            max_val = list_of_players[i].mu
            
    if max_val == 0:
        max_val = 1
    
    adjust_list_of_players = []
    
    for i in range(0, len(list_of_players)):
        adjust_list_of_players.append(env.create_rating(list_of_players[i].mu/max_val, (1/max_val)*list_of_players[i].phi))
        
    return adjust_list_of_players


# In[ ]:


def New_FFA_win_probability(list_of_players):
    list_of_win_probs = []
    
    #Looping through each player to calculate the integral (win probability) from their perspective
    for i in range(0, len(list_of_players)):
        def func_e(x, player):
            return math.exp((-(x-player.mu)**2)/(2*player.phi**2))

        def func_prod(x, player):
            return norm.cdf((x-player.mu)/player.phi)

        def func3(x, players):
            funcendelig = func_e(x, players[0])
            for i in range(1, len(players)):
                funcendelig = funcendelig*func_prod(x, players[i])
            return funcendelig

        val = 1/(math.sqrt(2*math.pi)*list_of_players[0].phi)
        integral_res = my_quad(func3, -np.inf, np.inf, midpoint=0, args=(list_of_players,)) #exact
        #integral_res = my_quad(func3, -3, 3, midpoint=0, args=(list_of_players,)) #estimated

        list_of_win_probs.append(val*integral_res[0])

        list_of_players = list_of_players[1:] + [list_of_players[0]]
        
    return list_of_win_probs


# In[ ]:


def qual_case(ratio):
    # A heuristic quality check (should be changed to match Glicko values)
    
    #ratio = no of players that HAVE NOT PLAYED MORE THAN 2 MATCHES AHEAD / no of players CLOSE TO PLAYER 1
    
    if ratio > 0.7:
        return 0.01
    elif ratio > 0.5:
        return 0.10
    elif ratio > 0.4:
        return 0.105
    else:
        return 0.145


# In[ ]:


def real_skill(dou, teamidFFA, n, backupRS):
    # Places the real skill as the 'mu' value in arrays; returns 'real skill' arrays for both teams, to be used with
    # win_probability method
    realFFA = np.empty(n, dtype=dict)
    
    for k in range(0, n):
        ind = teamidFFA[k]
        if ind == -1:
            realFFA[k] = env.create_rating(backupRS, 1, 0.06)
        for i in dou:
            if i == ind:
                realFFA[k] = env.create_rating(dou[i]['RS'], 1, 0.06)
    
    return realFFA


# In[ ]:


def setup_glicko_sim(amount, first_run):
    
    random_val1 = ran.randint(0,10000)
    random_val2 = ran.randint(0,10000)
    
    if first_run == True:
        np.random.seed(1120)
        ran.seed(8324)
    else:
        np.random.seed(random_val1)
        ran.seed(random_val2)
    
    norm_dist = np.random.normal(1500, 350, amount)

    dict_of_players = {}

    for i in range(0, amount):
        dict_of_players[i] = {}
        dict_of_players[i]['Rating'] = env.create_rating(1500, 350, 0.06) #Standard unrated player in Glicko
        dict_of_players[i]['Count'] = 0 #match count
        dict_of_players[i]['WinCount'] = 0 #win count
        dict_of_players[i]['TWR'] = [0.0] #tracked win ratio
        dict_of_players[i]['RS'] = norm_dist[i] #normal dist, assigned underlying real skill
        dict_of_players[i]['PointVector'] = []
        dict_of_players[i]['PlacementVector'] = []

    newdict = dict_of_players

    return newdict


# In[ ]:


def Glickomatch(dou, n, limgen, dummy_count, skill_band = 60):
    
    mu_s = skill_band
    
    p = np.empty(n, dtype=type)
    pid = np.empty(n, dtype=type)
    
    tdict = dou
    tdict = dict((k,v) for k,v in dou.items() if v['Count'] < limgen)
    
    doulen = len(tdict)
    ordering = np.random.choice(doulen, doulen, replace=False)
    
    temp1, temp2 = env.create_rating(1500, 350, 0.06), -1
    oe = []
    
    p[0], pid[0], oe = choose_player(tdict, ordering[0], temp1, temp2, oe)
    
    tdict2 = generatePoolCloseToPlayer(p[0], pid[0], tdict)
    doulen2 = len(tdict2)
    
    for i in range(1, n):
        if len(tdict2) == len(oe):
            p[i], pid[i], oe, dummy_count = find_player_dum(tdict, p[0].mu, p[0].phi, pid[0], 
                                                            temp1, temp2, oe, dummy_count, mu_s)
        else:
            p[i], pid[i], oe, dummy_count = find_player_dum(tdict2, p[0].mu, p[0].phi, pid[0], 
                                                            temp1, temp2, oe, dummy_count, mu_s)
    
    teamFFA = np.empty(n, dtype=type)
    teamidFFA = np.empty(n, dtype=type)
    
    for i in range(0, n):
        teamFFA[i] = p[i]
        teamidFFA[i] = pid[i]
        
    return teamFFA, teamidFFA, doulen2, dummy_count


# In[ ]:


def choose_player(dou, tbf, p, pid, oe):
    # check whether tbf (to be found) is in oe first; if yes, stop the function; if no, the player is added
    
    c = 0
    for i in dou:
        if tbf == c:
            if i in oe:
                break
            else:
                p = dou[i]['Rating']
                pid = i
                oe = oe + [pid]
        c = c + 1
    
    return p, pid, oe


# In[ ]:


def find_player_dum(dou, mu, phi, tbfid, p, pid, oe, dummy_count, mu_s = 60):
    # find a player to match the player with skill rating mu; if none is found, throw away both players
    
    if (len(dou) == 0 or len(dou) == len(oe)):
        p = env.create_rating(1500, 350, 0.06)
        pid = -1
        dummy_count += 1
        return p, pid, oe, dummy_count
    
    breakVar = 0
    while (breakVar < 7):
        for i in dou:
            if checkRatings(mu, phi, dou[i]['Rating'].mu, dou[i]['Rating'].phi) and not i in oe:
                p = dou[i]['Rating']
                pid = i
                oe = oe + [pid]
                return p, pid, oe, dummy_count
        mu_s = mu_s * 1.15
        breakVar = breakVar + 1
    
    for i in dou:
        p = dou[i]['Rating']
        pid = i
        oe = oe + [pid]
        return p, pid, oe, dummy_count


# In[ ]:


def generatePoolCloseToPlayer(player, playerid, temporarydict):
    tdict2 = {}
    
    playerValueTop = player.mu + player.phi
    playerValueBottom = player.mu - player.phi
    
    for i in temporarydict:
        
        topValue = temporarydict[i]['Rating'].mu + temporarydict[i]['Rating'].phi
        bottomValue = temporarydict[i]['Rating'].mu - temporarydict[i]['Rating'].phi
        
        if not i == playerid:
            if not playerValueTop < bottomValue and not topValue < playerValueBottom:
                tdict2[i] = temporarydict[i]
    
    return tdict2


# In[ ]:


def checkRatings(mu, phi, altmu, altphi):
    playerValueTop = mu + phi
    playerValueBottom = mu - phi
    
    topValue = altmu + altphi
    bottomValue = altmu - altphi
    
    if not playerValueTop < bottomValue and not topValue < playerValueBottom:
        return True
    else:
        return False


# In[ ]:


def quality_check(env, playerRatings, n):
    maxEx1 = 0
    maxEx2 = 0
    
    for i in range(0,n):
        for j in range(0,n):
            if not i == j:
                expected_score1 = env.expect_score(playerRatings[i], playerRatings[j], env.reduce_impact(playerRatings[j]))
                expected_score2 = env.expect_score(playerRatings[j], playerRatings[i], env.reduce_impact(playerRatings[i]))
                if expected_score1 > maxEx1:
                    maxEx1 = expected_score1
                    maxEx2 = expected_score2
    
    return abs(maxEx1 - maxEx2) / 4


# In[ ]:


def NewMatchResults(realFFA, teamidFFA):
    teamidTemp = list(teamidFFA)
    resultList = []
    n = len(realFFA)
    
    adjustRealTemp = adjust_rating(realFFA)
    
    while n >= 2:
        
        FFAWinArray = New_FFA_win_probability(adjustRealTemp)
        winIndex = ran.choices(teamidTemp,weights=FFAWinArray,k=1)
        resultList.append(winIndex[0])
        adjustRealTemp.pop(teamidTemp.index(winIndex[0]))
        teamidTemp.remove(winIndex[0])
        
        n = n - 1
        
    resultList.append(teamidTemp[0])
        
    return resultList


# In[ ]:


def NewMatchResultsWithRandomLosers(realFFA, teamidFFA):
    
    teamidTemp = list(teamidFFA)
    resultList = []
    n = len(realFFA)
        
    adjustRealTemp = adjust_rating(realFFA)
    FFAWinArray = New_FFA_win_probability(adjustRealTemp)
    winIndex = ran.choices(teamidTemp,weights=FFAWinArray,k=1)
    resultList.append(winIndex[0])
    adjustRealTemp.pop(teamidTemp.index(winIndex[0]))
    teamidTemp.remove(winIndex[0])

    n = n - 1
    
    while n > 1:
        
        winIndex = ran.choices(teamidTemp,k=1)
        resultList.append(winIndex[0])
        adjustRealTemp.pop(teamidTemp.index(winIndex[0]))
        teamidTemp.remove(winIndex[0])
        
        n = n - 1
        
    resultList.append(teamidTemp[0])
        
    return resultList


# In[ ]:


def PairwiseMethod(dou, resultList):

    pointWinner = drawPointsForWinner(1)
    pointSecond = drawPointsForSecond(1)
    pointThird = drawPointsForThird(1, pointSecond[0])
    pointFourth = drawPointsForFourth(1, pointThird[0])
    pointList = pointWinner + pointSecond + pointThird + pointFourth
    pointList.sort(reverse=True)
    
    ratioPoints = [pointList[0]/(pointList[0]+pointList[1])]
    for i in range(1,len(resultList)):
        ratioPoints = ratioPoints + [pointList[i]/(pointList[i] + pointList[i+1])]
        if i+2 == len(resultList):
            ratioPoints = ratioPoints + [pointList[i]/(pointList[i] + pointList[i+1])]
            break
    
    r = []
            
    for i in range(0,len(resultList)):
        if not resultList[i] == -1:
            r = r + [dou[resultList[i]]['Rating']]
        else:
            r = r + [env.create_rating(1500, 350, 0.06)]
    
    rated_a = env.rate(r[0], [(WIN, r[1])])
    rated_b = env.rate(r[1], [(LOSS, r[0])])
    r[0] = env.create_rating(abs(rated_a.mu-r[0].mu) + r[0].mu, rated_a.phi, rated_a.sigma)
    
    for i in range(1, len(resultList)):
        if pointList[i] == pointList[i+1]:
            rated_a = env.rate(rated_b, [(DRAW, r[i+1])])
            rated_b = env.rate(r[i+1], [(DRAW, r[i])])
        else:
            rated_a = env.rate(rated_b, [(WIN, r[i+1])])
            rated_b = env.rate(r[i+1], [(LOSS, r[i])])
        
        if i+2 == len(resultList):
            r[i] = env.create_rating(r[i].mu - abs(rated_a.mu-r[i].mu), rated_a.phi, rated_a.sigma)
            r[i+1] = env.create_rating(r[i+1].mu - abs(rated_b.mu-r[i+1].mu), rated_b.phi, rated_b.sigma)
            break
        else:
            r[i] = env.create_rating(abs(rated_a.mu-r[i].mu) + r[i].mu, rated_a.phi, rated_a.sigma)
    
    for i in range(0, len(resultList)):
        if not resultList[i] == -1:
            if i == 0:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['WinCount'] = dou[resultList[i]]['WinCount'] + 1
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]
                
            else:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]
    return


# In[ ]:


def DesignRatedList(pointList, r, altMethod = True):
    range_number = len(pointList)
    
    winLossMatrix = []

    a = np.ones((range_number,range_number))
    arr = np.triu(a)

    for i in range(1, range_number-1):
        for j in range(i, range_number-1):
            if not pointList[j] == pointList[j+1]:
                for k in range(j+1, range_number):
                    arr[i][k] = 1
                    arr[k][i] = 0
                break
            else:
                arr[i][j+1] = 0.5
                arr[j+1][i] = 0.5

    ivariable = 1
    if altMethod:
        ivariable = 0
                
    for i in range(0, len(r) - ivariable):
        winLossList = []

        for j in range(0, len(r)):
            if not i == j:
                if arr[i][j] == 1:
                    winLossList.append((WIN, r[j]))
                elif arr[i][j] == 0:
                    winLossList.append((LOSS, r[j]))
                else:
                    winLossList.append((DRAW, r[j]))

        winLossMatrix.append(winLossList)
        
    if not altMethod:
        winLossList = [(LOSS, r[0])]
        winLossMatrix.append(winLossList)
        
    ratedList = []

    for m in range(0, len(winLossMatrix)):
        ratedList.append(env.rate(r[m], winLossMatrix[m]))
    
    return ratedList


# In[ ]:


def StrictMethod(dou, resultList):

    # NB! - This method only works for even number of FFA players
    
    pointWinner = drawPointsForWinner(1)
    pointSecond = drawPointsForSecond(1)
    pointThird = drawPointsForThird(1, pointSecond[0])
    pointFourth = drawPointsForFourth(1, pointThird[0])
    pointList = pointWinner + pointSecond + pointThird + pointFourth
    pointList.sort(reverse=True)
    
    ratioPoints = [pointList[0]/(pointList[0]+pointList[1])]
    for i in range(1,len(resultList)):
        ratioPoints = ratioPoints + [pointList[i]/(pointList[i] + pointList[i+1])]
        if i+2 == len(resultList):
            ratioPoints = ratioPoints + [pointList[i]/(pointList[i] + pointList[i+1])]
            break

    r = []
            
    for i in range(0, len(resultList)):
        if not resultList[i] == -1:
            r = r + [dou[resultList[i]]['Rating']]
        else:
            r = r + [env.create_rating(1500, 350, 0.06)]
    
    winLossMatrix = []
    lossCount = 0
    winCount = len(r)-1
    count = 1
    range_number = len(resultList)-1

    ratedList = DesignRatedList(pointList, r) #alt method
    #ratedList = DesignRatedList(pointList, r, False) # False = doesn't use alt method
    
    for n in range(0, len(r)):
        r[n] = env.create_rating(ratedList[n].mu, ratedList[n].phi, ratedList[n].sigma)
    
    for i in range(0, len(resultList)):
        if not resultList[i] == -1:
            if i == 0:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['WinCount'] = dou[resultList[i]]['WinCount'] + 1
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]
                
            else:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]
                
    return


# In[ ]:


def GapsMethod(dou, resultList):

    pointWinner = drawPointsForWinner(1)
    pointSecond = drawPointsForSecond(1)
    pointThird = drawPointsForThird(1, pointSecond[0])
    pointFourth = drawPointsForFourth(1, pointThird[0])
    pointList = pointWinner + pointSecond + pointThird + pointFourth
    pointList.sort(reverse=True)
    
    pointGap = 0

    for i in range(1, len(pointList)):
        pointGap += pointList[0]-pointList[i]

    ratioPointGap = []
    for i in range(1, len(pointList)):
        ratioPointGap = ratioPointGap + [(pointList[0]-pointList[i])/pointGap]
    
    r = []
            
    for i in range(0,len(resultList)):
        if not resultList[i] == -1:
            r = r + [dou[resultList[i]]['Rating']]
        else:
            r = r + [env.create_rating(1500, 350, 0.06)]
    
    winList = []
    winLossArr = []

    for i in range(1, len(r)):
        winList.append((WIN, r[i]))
    winLossArr.append(winList)

    lossList = []
    for i in range(1, len(r)):
        lossList = []
        lossList.append((LOSS, r[0]))
        winLossArr.append(lossList)

    ratedList = []

    for i in range(0, len(r)):
        ratedList.append(env.rate(r[i], winLossArr[i]))

    for i in range(0, len(r)):
        if i == 0:
            rated1_2 = env.rate(r[i], [(WIN, r[1])])
            winner_rating = env.create_rating(r[i].mu+abs(rated1_2.mu-r[i].mu)*ratioPointGap[0], 
                                              rated1_2.phi, rated1_2.sigma)
            
            rated1_3 = env.rate(r[i], [(WIN, r[2])])
            winner_rating = env.create_rating(winner_rating.mu+abs(rated1_3.mu-r[i].mu)*ratioPointGap[1], 
                                              rated1_2.phi, rated1_2.sigma)
            
            rated1_4 = env.rate(r[i], [(WIN, r[3])])
            winner_rating = env.create_rating(winner_rating.mu+abs(rated1_4.mu-r[i].mu)*ratioPointGap[2], 
                                              rated1_2.phi, rated1_2.sigma)
            
        else:
            r[i] = env.create_rating(r[i].mu-abs(ratedList[i].mu-r[i].mu)*ratioPointGap[i-1], 
                                     ratedList[i].phi, ratedList[i].sigma)
            
    r[0] = winner_rating
    
    for i in range(0, len(resultList)):
        if not resultList[i] == -1:
            if i == 0:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['WinCount'] = dou[resultList[i]]['WinCount'] + 1
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]
                
            else:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]
    return


# In[ ]:


def CasualMethod(dou, resultList):
    # Currently only works for the 4-player FFA case
    
    pointWinner = drawPointsForWinner(1)
    pointSecond = drawPointsForSecond(1)
    pointThird = drawPointsForThird(1, pointSecond[0])
    pointFourth = drawPointsForFourth(1, pointThird[0])
    pointList = pointWinner + pointSecond + pointThird + pointFourth
    pointList.sort(reverse=True)
    
    r = []
            
    for i in range(0,len(resultList)):
        if not resultList[i] == -1:
            r = r + [dou[resultList[i]]['Rating']]
        else:
            r = r + [env.create_rating(1500, 350, 0.06)]
    
    rated1 = env.rate(r[0], [(WIN, r[1]), (WIN, r[2]), (WIN, r[3])])
    rated2 = env.rate(r[1], [(LOSS, r[0]), (DRAW, r[2]), (DRAW, r[3])])
    rated3 = env.rate(r[2], [(LOSS, r[0]), (DRAW, r[1]), (DRAW, r[3])])
    rated4 = env.rate(r[3], [(LOSS, r[0]), (DRAW, r[1]), (DRAW, r[2])])

    r[0] = env.create_rating(rated1.mu, rated1.phi, rated1.sigma)
    r[1] = env.create_rating(rated2.mu, rated2.phi, rated2.sigma)
    r[2] = env.create_rating(rated3.mu, rated3.phi, rated3.sigma)
    r[3] = env.create_rating(rated4.mu, rated4.phi, rated4.sigma)
    
    for i in range(0, len(resultList)):
        if not resultList[i] == -1:
            if i == 0:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['WinCount'] = dou[resultList[i]]['WinCount'] + 1
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]
                
            else:
                dou[resultList[i]]['Count'] = dou[resultList[i]]['Count'] + 1
                dou[resultList[i]]['Rating'] = r[i]
                dou[resultList[i]]['TWR'] = dou[resultList[i]]['TWR'] +                                             [dou[resultList[i]]['WinCount']/dou[resultList[i]]['Count']]
                dou[resultList[i]]['PlacementVector'] = dou[resultList[i]]['PlacementVector'] + [i+1]
                dou[resultList[i]]['PointVector'] = dou[resultList[i]]['PointVector'] + [pointList[i]]                
    return


# In[ ]:


def RunGlickoFFASim(newdict, compMatch, gamesUntilPrint):
    # We calculate the probability of P1 winning, then P2, then P3 etc. by using their "real" skill ratings
    # We stop when at least one player has played compMatch games or 100000 matches total have played out (failsafe)

    below = True # Keeps the matches going while compMatch has not been reached
    breakWhile = 0 # Failsafe for infinite loop
    limgen = 2 # Generational gap allowed
    internalCount = 0 # Counts how many failed matches
    internalBreak = 0 # Allows a constant lax in quality per failed match happening
    internalSlope = 0.01 # Amount of constant lax
    internalAllow = 10 # Allow 10 matches before counting up
    
    dummy_player_count = 0

    x = [] # time series of number of loops
    y = [] # time series of uncertainty (sigma)
    z = [] # time series of pearson correlation between RS and TS
    a = [] # time series of spearman correlation between RS and TS
    b = [] # time series of NDCG score between RS and TS
    
    w = [] # keep an eye on the quality of various matches

    myvec1 = [] # time series of mus chosen for Team 1
    myvec2 = [] # time series of mus chosen for Team 2
    vavec1 = [] # time series of sigmas chosen for Team 1
    vavec2 = [] # time series of sigmas chosen for Team 2

    n = 4

    start = time.time()

    while below and breakWhile < 1000000:

        truS = []
        estS = []

        # The matching module - we match based on their "estimated" skill rating
        teamFFA, teamidFFA, dictlength, dummy_player_count = Glickomatch(newdict, n, limgen, dummy_player_count)

        ratio = dictlength / len(newdict)
        
        # We find the "real" skill ratings for the players in the match - to be used for finding the winner
        realFFA = real_skill(newdict, teamidFFA, n, newdict[teamidFFA[0]]['RS'])

        if quality_check(env, teamFFA, n) - internalBreak*internalSlope < qual_case(ratio):
            internalBreak = 0
            internalCount = 0
            
            # ------------------------------------------------------------
            # In this section, enable the ordering and the method you wish to run but commenting appropriately
            # ------------------------------------------------------------
            
            #New MatchResults methods - uses integrals to calculate the results:
            
            # Strong ordering
            resultList = NewMatchResults(realFFA, teamidFFA)
            
            # Weak ordering
            #resultList = NewMatchResultsWithRandomLosers(realFFA, teamidFFA)
            
            # ------------------------------------------------------------
            # Here you choose which skill rating update method you wish to use
            # ------------------------------------------------------------
            
            # Method 1 - Pairwise
            PairwiseMethod(newdict, resultList)
            
            # Method 2 - Gaps method
            #GapsMethod(newdict, resultList)
            
            # Method 3 - Strict
            #StrictMethod(newdict, resultList)
            
            # Method 4 - Casual
            #CasualMethod(newdict, resultList)

            cArr = [] # count array

            below = False
            lowgen = 20000 # Find the lowest generation player
            for i in newdict:
                if newdict[i]['Count'] < lowgen:
                    lowgen = newdict[i]['Count']

                cArr = cArr + [newdict[i]['Count']]

                if newdict[i]['Count'] < compMatch:
                    below = True # If any player has not played compMatch amount, then keep it going

                if breakWhile % gamesUntilPrint == 0:
                    # true and estimated skill every gamesUntilPrint times

                    truS = truS + [newdict[i]['RS']]  
                    estS = estS + [newdict[i]['Rating'].mu]

            limgen = lowgen + 2

            if breakWhile % gamesUntilPrint == 0:
                z = z + [pearsonr(truS, estS)] # Calculating correlation every gamesUntilPrint times
                a = a + [spearmanr(truS, estS)]
                b = b + [ndcg_score([truS], [estS])]
                y = y + [newdict[i]['Rating'].phi]
                x = x + [breakWhile]

            breakWhile = breakWhile + 1
        else:
            internalCount = internalCount + 1
            if internalCount >= internalAllow:
                internalBreak = internalBreak + 1

    slut = time.time()
    print("Time it took: " + str(slut - start))
    
    return x, y, z, a, b, dummy_player_count


# In[ ]:


def perform_spearman_wilcoxon_ndcg(dict_of_players):
    list_RS = []
    list_ES = []
    list_D = []
    for i in dict_of_players:
        list_RS = list_RS + [dict_of_players[i]['RealRanking']]
        list_ES = list_ES + [dict_of_players[i]['EstRanking']]
        list_D = list_D + [dict_of_players[i]['RealRanking'] - dict_of_players[i]['EstRanking']]
    spear = spearmanr(list_RS, list_ES)
    wil = wilcoxon(list_D)
    wil2 = 'N/A'
    wil3 = 'N/A'
    if wil[1] <= 0.05:
        wil2 = wilcoxon(list_D, alternative='less')
        wil3 = wilcoxon(list_D, alternative='greater')
    
    ndcg = ndcg_score([list_RS], [list_ES])
    
    return list_RS, list_ES, list_D, spear, wil, wil2, wil3, ndcg


# In[ ]:


def assign_ranking(dic, amount):
    realplace = amount
    estplace = amount
    for i in range(0,amount):
        lowest = 15000
        lowest_j = -1
        for j in dic:
            if dic[j]['RealRanking'] == -1 and dic[j]['RS'] < lowest:
                lowest = dic[j]['RS']
                lowest_j = j
        dic[lowest_j]['RealRanking'] = realplace
        realplace = realplace - 1
        lowest = 15000
        lowest_j = -1
        for j in dic:
            if dic[j]['EstRanking'] == -1 and dic[j]['Rating'].mu < lowest:
                lowest = dic[j]['Rating'].mu
                lowest_j = j
        dic[lowest_j]['EstRanking'] = estplace
        estplace = estplace - 1
    
    return dic


# In[ ]:


def calculate_mean_median(dict_of_players):
    list_RS = []
    list_ES = []
    for i in dict_of_players:
        list_RS = list_RS + [dict_of_players[i]["RS"]]
        list_ES = list_ES + [dict_of_players[i]["Rating"].mu]
    mean_RS = np.mean(list_RS)
    mean_ES = np.mean(list_ES)
    med_RS = np.median(list_RS)
    med_ES = np.median(list_ES)
    var_RS = np.var(list_RS)
    var_ES = np.var(list_ES)
    return mean_ES, med_ES, var_ES, mean_RS, med_RS, var_RS


# In[ ]:


env = Glicko2(tau=0.3) #Set between 0.3 and 1.2 for best results depending on game

amount = player_amount

dummyplayercountList = []
matchcountList = []

pearsonList = []
spearmanList = []
ndcgList = []

for i in range(0, sim_amount):
    if i == 0:
        newdict = setup_glicko_sim(amount, True)        
    else:
        newdict = setup_glicko_sim(amount, False)

    x, y, z, a, b, dummy_player_count = RunGlickoFFASim(newdict, match_amount, gamesUntilPrint)

    print("I just finished sim number: " + str(i+1))
    print()

    corrdict = {'x' : x, 'y' : y, 'z' : z, 'a' : a, 'b' : b}

    dummyplayercountList.append(dummy_player_count)

    zz = []
    der = 0
    for i in range(0, len(z)):
        zz = zz + [z[i][0]]
        if (i == len(z)-1):
            der = z[i][0]
            
    aa = []
    der = 0
    for i in range(0, len(a)):
        aa = aa + [a[i][0]]
        if (i == len(a)-1):
            der = a[i][0]

    pearsonList.append(zz)
    spearmanList.append(aa)
    ndcgList.append(b)

    matchcount = 0

    for i in newdict:
        newdict[i]['RealRanking'] = -1
        newdict[i]['EstRanking'] = -1
        matchcount += newdict[i]["Count"]
    newdict = assign_ranking(newdict, amount)

    matchcountList.append(matchcount)


# In[ ]:


header = [f'Sim {i+1}' for i in range(sim_amount)]

pearson_df = pd.DataFrame(pearsonList).T
spearman_df = pd.DataFrame(spearmanList).T
ndcg_df = pd.DataFrame(ndcgList).T

pearson_df.columns = header
spearman_df.columns = header
ndcg_df.columns = header

pearson_df.to_csv('Pearson Data.csv', index=False)
spearman_df.to_csv('Spearman Data.csv', index=False)
ndcg_df.to_csv('NDCG Data.csv', index=False)


# In[ ]:





# In[ ]:




