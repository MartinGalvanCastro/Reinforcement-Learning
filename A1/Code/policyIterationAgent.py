# policyIterationAgent.py
# ------------------------------------------------

import mdp
import util
import random
from pprint import pprint
from learningAgents import ValueEstimationAgent


class PolicyIterationAgent(ValueEstimationAgent):
    """
        Policy Iteration Agent Class

        It is an implementation of ValueEstimation Agent, 
        for getting the parameters for initialization


        A POliccyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs Policcy iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your Policy iteration agent should take an mdp on
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
        self.policies = util.Counter()  # Best Policy Dict
        self.runPolicyIteration()

    def runPolicyIteration(self):
        # Policy Iteration Code here

        # Initialize Random Policies
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0  # Initialize Values
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(
                    state)  # Get all actions of State S
                policy = random.choice(
                    actions)  # Choose one as starting policy
                self.policies[state] = policy
            else:
                self.policies[state] = "NaN"

        # Do Policy Iteration for each state
        for _ in range(self.iterations):
            for state in states:
                if not self.mdp.isTerminal(state):
                    self.runPolicyEvaluation(state)
                    self.runPolicyImprovement(state)

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

    def runPolicyEvaluation(self, state):
        """
        Policy Evaluation Step
        """
        policy = self.policies[state]  # Get the policy of the state
        value = 0
        # Get Q-Value of the policy
        for nextState in self.mdp.getTransitionStatesAndProbs(state, policy):
            prob = nextState[1]
            reward = self.mdp.getReward(state, policy, nextState[0])
            oldValue = self.values[nextState[0]]
            value += prob*(reward+self.discount*oldValue)
        self.values[state] = value  # Update Value

    def runPolicyImprovement(self, state):
        """
        Policy Improvement Step
        """
        # Initialize variables
        actions = self.mdp.getPossibleActions(state)
        maxQVal = float("-inf")
        bestPolicy = None

        # Iterate over all actions
        for action in actions:
            value = 0
            # Calculate Q(s,a)
            for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
                prob = nextState[1]
                reward = self.mdp.getReward(state, action, nextState[0])
                oldValue = self.values[nextState[0]]
                value += prob*(reward+self.discount*oldValue)

            # Check max value for best policy
            if value > maxQVal:
                maxQVal = value
                bestPolicy = action

        self.policies[state] = bestPolicy  # Update Value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
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
