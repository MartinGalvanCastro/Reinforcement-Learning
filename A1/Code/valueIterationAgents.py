# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import mdp
import util
from pprint import pprint

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        valuesAtK = util.Counter()  # Values at iteration K
        valuesAtK_1 = util.Counter()  # Values at iteration K-1
        states = self.mdp.getStates()  # States

        # First Iteration, initialize everyting
        for state in states:
            valuesAtK_1[state] = 0

        # Iterations when K>=1
        for _ in range(1, self.iterations+1):

            # Iterate over all states
            for state in states:
                actions = self.mdp.getPossibleActions(
                    state)  # Get all actions of the state
                maxQVal = float("-inf")  # Declare the min float
                # print("---------------------------------------------")
                #pprint(f"State: {state}")
                # Iterate over all actions
                for action in actions:
                    #pprint(f"Action: {action}")
                    Qvalue = 0  # Q value of action a
                    nextStates = self.mdp.getTransitionStatesAndProbs(
                        state, action)  # Get all output states
                    # Iterate over output states
                    for nextState in nextStates:
                        stateP = nextState[0]  # Get State Id
                        prob = nextState[1]  # Get state prob
                        reward = self.mdp.getReward(
                            state, action, stateP)  # Get rewards
                        pastValueStateP = valuesAtK_1[stateP]  # Get past value
                        # pprint(
                        # f"StatesP:  {nextState} // Reward: {reward} // PastValue: {pastValueStateP}")
                        Qvalue += prob * \
                            (reward + self.discount*pastValueStateP)

                    maxQVal = max(maxQVal, Qvalue)  # Update maxQVal

                # Double check maxQVal is not -inf
                if not maxQVal > float("-inf"):
                    # Fix
                    maxQVal = 0

                # Save QVal for state S
                valuesAtK[state] = maxQVal

            # Values at K will pass to be values at K-1 for next iteration
            # pprint("-------------------------------------------------")
            #pprint(f"Values-state: {valuesAtK}")
            valuesAtK_1 = valuesAtK
            valuesAtK = util.Counter()

        self.values = valuesAtK_1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0  # Q value = Q(s,a)
        nextStates = self.mdp.getTransitionStatesAndProbs(
            state, action)  # Output states

        # Iterate over all output states
        for nextState in nextStates:
            stateP = nextState[0]  # State Id
            prob = nextState[1]  # Transition probability
            reward = self.mdp.getReward(
                state, action, stateP)  # Transition Reward
            v = self.values[stateP]  # Old value of transition
            value += prob * (reward + self.discount * v)  # Sum of Q values
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # Initialize variables
        bestAction = None
        bestQValue = float('-infinity')
        actions = self.mdp.getPossibleActions(state)

        # Iterate over all actions
        for action in actions:
            qValue = self.getQValue(state, action)  # Get Q(s,a)

            # Check if qValue is greater
            if qValue > bestQValue:
                # Update
                bestAction = action
                bestQValue = qValue

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()  # States
        for k in range(self.iterations):
            # Mod was used to loop trough the list
            current_state = states[k % len(states)]
            if not self.mdp.isTerminal(current_state):
                actions = self.mdp.getPossibleActions(current_state)
                maxQVal = float("-inf")
                for action in actions:
                    Qval = 0
                    nextStates = self.mdp.getTransitionStatesAndProbs(
                        current_state, action)
                    for nextState in nextStates:
                        stateP = nextState[0]
                        prob = nextState[1]
                        reward = self.mdp.getReward(
                            current_state, action, stateP)
                        pastValueStateP = self.values[stateP]
                        Qval += prob * \
                            (reward + self.discount*pastValueStateP)
                    maxQVal = max(maxQVal, Qval)
                if not maxQVal > float("-inf"):
                    maxQVal = 0
                self.values[current_state] = maxQVal


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        priority_Queue = util.PriorityQueue()  # Priority Queue
        states = self.mdp.getStates()  # All states
        predecessors = {}  # Dict for predecessors state

        for state in states:  # Initialize predecessors dict
            predecessors[state] = set()

        for state in states:
            for action in self.mdp.getPossibleActions(state):
                nextStates = self.mdp.getTransitionStatesAndProbs(
                    state, action)
                for stateP in nextStates:
                    predecessors[stateP[0]].add(state)

        for state in states:
            if not self.mdp.isTerminal(state):
                bestAction = self.computeActionFromValues(state)
                maxQValue = self.computeQValueFromValues(state, bestAction)
                diff = abs(self.values[state]-maxQValue)
                priority_Queue.push(state, -diff)

        for _ in range(self.iterations):
            if priority_Queue.isEmpty():
                return
            s = priority_Queue.pop()

            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                maxVal = float('-infinity')
                for action in actions:
                    value = 0
                    nextStateInfos = self.mdp.getTransitionStatesAndProbs(
                        s, action)
                    for nextStateInfo in nextStateInfos:
                        nextState = nextStateInfo[0]
                        prob = nextStateInfo[1]
                        reward = self.mdp.getReward(s, action, nextState)
                        v = self.values[nextState]
                        value += prob * (reward + self.discount * v)
                    maxVal = max(maxVal, value)
                if maxVal > float('-infinity'):
                    self.values[s] = maxVal
                else:
                    self.values[s] = 0

            for p in predecessors[s]:
                bestAction = self.computeActionFromValues(p)
                if bestAction == None:
                    continue
                highestQValue = self.computeQValueFromValues(p, bestAction)
                diff = abs(highestQValue - self.values[p])

                if diff > self.theta:
                    priority_Queue.update(p, -diff)
