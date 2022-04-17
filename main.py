""""
Programming Assignment 3: Reinforcement Learning
Date: 4/09/2022
Author: Inigo Hohmeyer
"""
#   For the first line:
#   The first has the number of non terminal states
#   The second is the number of terminal states
#   The third is the number of rounds
#   The fourth is the frequency for which to print
#   The fifth is the explore/exploit tradeoff

#   For the second line:
#   Alternates between the state number of the terminal state and the corresponding state number

#   For the third line:
#   The remaining lines will specify the transition probabilities between the states and the corresponding actions
#   e.g. (state: action state transition probability to that state)

#  We will go through the states randomly and then over time we will learn which actions are the best to choose
#  the best actions.


#   For the states we can just use dictionaries


#   The input reader takes the input and puts all the states into a dictionary
#   with order [state][action][nextState] = probability of going to that next state
import copy
import math
import random
from random import randint


def InputReader(input):
    stateDict = {}
    with open(input, "r", encoding='utf-8') as file:
        file = file.readlines()
        #   This converts the first line to an array of ints
        firstLine = list(map(int, file[0].split()))
        #   gathers the M, frequency, and rounds
        rounds = firstLine[2]
        frequency = firstLine[3]
        parameterM = firstLine[4]
        termStateDict = {}
        prevValue = 0
        for index, value in enumerate(file[1].split()):
            if index % 2 == 0:
                termStateDict[float(value)] = {}
                prevValue = float(value)
            else:
                termStateDict[prevValue] = float(value)
        for i in file[2:]:
            stateAction = i.split()
            state = int(stateAction[0].split(":")[0])
            action = int(stateAction[0].split(":")[1])
            if state not in stateDict:
                stateDict[state] = {}
                stateDict[state][action] = {}
            else:
                stateDict[state][action] = {}

            for index, value in enumerate(stateAction[1:]):
                if index % 2 == 0:
                    stateDict[state][action][float(value)] = {}
                    prevState = float(value)
                else:
                    stateDict[state][action][prevState] = float(value)
    return stateDict, termStateDict, rounds, frequency, parameterM

    #   To do the process we will first start at any of the none terminal states
def GradDesc(stateDict, termStateDict, rounds, v, M):
    top = max(termStateDict.values())
    bottom = min(termStateDict.values())
    count = learn_maker(stateDict)
    total = learn_maker(stateDict)
    counter = 0
    for i in range(rounds):
        currentState = randint(min(stateDict.keys()), max(stateDict.keys()))
        while currentState not in termStateDict:
            #   in each round the count is set to 0
            currentCount = learn_maker(stateDict)
            #   action is chosen randomly
            action = chooseAction(currentState, count, total, M, top, bottom)
            #   it's only incremented if it's 0
            if currentCount[currentState][action] == 0:
                currentCount[currentState][action] = 1
            #   next is chosen from the action using the
            #   random distribution function
            next = randDist(stateDict[currentState][action])
            #   sets the current state to the next state
            currentState = next
        #   Goes through the current count
        for i in currentCount:
            for j in currentCount[i]:
                #   if it's set to 1
                #   then the overall count is incremented by 1
                if currentCount[i][j] == 1:
                    count[i][j] += 1
                    #   the total then adds the value for the ending reward
                    total[i][j] += termStateDict[currentState]
        #
        if v == 0:
            continue
        #   Adds a round after a round is complete
        counter += 1
        #   If the number of rounds is divisible by the frequency then we print
        if counter % v == 0:
            printTotalCount(count, total, counter, "output.txt")
    #   If the frequency is
    if v == 0:
        printTotalCount(count, total, counter, "output.txt")


#   This function creates the tables which will store both the
#   Total and the Count
def learn_maker(stateDict):
    learnTable = {}
    for i in stateDict:
        learnTable[i] = {}
        for j in stateDict[i]:
            learnTable[i][j] = 0
    return learnTable


#   This function goes through the count and the total rounds and prints
#   them out for each state and action
#   The best functions will also be calculated
#   for each state. And these will be put for bestaction[i]
#   however if count is 0. Then we will put u.
def printTotalCount(count, total, round, output):
    bestaction = {}
    print("\n")
    print(f"After {round} rounds")
    print("Count:")
    for i in count:
        print("\n")
        bestaction[i] = 0
        for j in count[i]:
            #  If there's any action in this state which
            #  has not been taken.
            #  Bestaction is set to "u"
            if count[i][j] == 0:
                bestaction[i] = "u"
            #   Otherwise if bestaction was already set to 0
            #   Then we will print the count
            elif bestaction[i] == "u":
                print(f"[{i}, {j}] = {count[i][j]}", end=" ")
                continue
            #   Otherwise if it's non zero and it's bigger than the current best action.
            #   we will set it to current bestaction
            elif total[i][j] / count[i][j] > bestaction[i]:
                bestaction[i] = j
            print(f"[{i}, {j}] = {count[i][j]}", end=" ")
    print("\n")
    print("Total:")
    for i in total:
        print("\n")
        for j in total[i]:
            print(f"[{i}, {j}] = {total[i][j]}", end=" ")
    print("\n")
    print("Best Action:", end=" ")
    for i in bestaction:
        print(f'{i}:{bestaction[i]}', end=" ")


def chooseAction(state, count, total, M, top, bottom):
    avg = {}
    #   Will choose any action which has been unexplored
    #  If we find an action in the state which has a count total of 0.
    #  then this is the first one that we return
    for i in count[state]:
        if count[state][i] == 0:
            return i
    #   Goes through and finds the average payoff of each action
    for i in count[state]:
        avg[i] = total[state][i] / count[state][i]
    #   Scales the average
    for i in avg:
        avg[i] = 0.25 + 0.75 * (avg[i] - bottom) / (top - bottom)
    #   Gets the value of c
    c = sum(count[state].values())
    up = {}
    for i in avg:
        up[i] = avg[i] ** (c / M)
    norm = sum(up.values())
    p = {}
    for i in up:
        p[i] = up[i] / norm
    return randDist(p)


#   We assume that probArray is a dictionary
def randDist(probArray):
    newProb = {}
    prev = 0
    for i in probArray:
        newProb[i] = probArray[i] + prev
        prev = newProb[i]
    x = random.uniform(0, 1)
    for i in newProb:
        if newProb[i] > x:
            return i


stateDict = InputReader("input.txt")[0]
termStateDict = InputReader("input.txt")[1]
rounds = InputReader("input.txt")[2]
frequency = InputReader("input.txt")[3]
M = InputReader("input.txt")[4]
print(stateDict)
print(learn_maker(stateDict))
GradDesc(stateDict, termStateDict, rounds, frequency, M)
