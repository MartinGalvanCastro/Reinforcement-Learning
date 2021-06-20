import random
from optparse import OptionParser


class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


oldValues = None
policies = None

parser = OptionParser()
parser.add_option("-p",
                  "--policy",
                  default="1",
                  type="int",
                  dest="policy",
                  help="1 For testing an arbitriary policcly. 2 For testing always fast policy. 3 For testing always slow policy")
(options, args) = parser.parse_args()
arg = options.policy

# State and Actions, discount and tolerance definition
S = ["Cool", "Warm", "Overheated"]
A = ["Fast", "Slow"]
gamma = 0.5
delta = 0.005

# Transition & Reward Function
T = {}
R = {}

newValues = {x: 0 for x in S}
convergence = {x: False for x in S}


# Actions from states
actions = {}
actions[S[0]] = [A[0], A[1]]
actions[S[1]] = [A[0], A[1]]
actions[S[2]] = []


if arg == 1:
    print(f"{bcolors.WARNING}It is going to be used random policies{bcolors.RESET}")
    oldValues = {x: 0 for x in S}
    policies = {x: random.choice(actions[x]) for x in S if len(actions[x]) > 0}
    policies[S[2]] = 0
elif arg == 2:
    print(f"{bcolors.WARNING}It is going to be used only fast policy{bcolors.RESET}")
    oldValues = {S[0]: -2/3, S[1]: -10, S[2]: 0}
    policies = {x: A[0] for x in S if len(actions[x]) > 0}
elif arg == 3:
    print(
        f"{bcolors.WARNING}It is going to be used always slow policy{bcolors.RESET}")
    oldValues = {S[0]: 2, S[1]: 2, S[2]: 0}
    policies = {x: A[1] for x in S if len(actions[x]) > 0}
else:
    print(f"{bcolors.FAIL}Option not valid, please run 'python L3_POlicyIteration.py -h' {bcolors.RESET}")
    exit("")

print(f"{bcolors.WARNING}Initial Policies: {policies}{bcolors.RESET}")
print(f"{bcolors.WARNING}Initial Values: {oldValues}{bcolors.RESET}")
print("----------------------------------------")

# Rewards and Transition of Cool State from any State
# T(C,F,C)=0.5 & T(C,F,W)=0.5
T[(S[0], A[0])] = [(S[0], 0.5), (S[1], 0.5)]

# R(C,F,C)=2 & R(C,F,W)=2
R[(S[0], A[0], S[0])] = 2
R[(S[0], A[0], S[1])] = 2

# T(C,S,C)=1 & R(C,S,C)=1
T[(S[0], A[1])] = [(S[0], 1.0)]
R[(S[0], A[1], S[0])] = 1.0

# Rewards and Transition of Warm State from any State
# T(W,F,O)=1 & R(W,F,O)=-10
T[(S[1], A[0])] = [(S[2], 1.0)]
R[(S[1], A[0], S[2])] = -10.0

# T(W,S,W)=0.5 & T(W,S,C)=0.5
T[(S[1], A[1])] = [(S[0], 0.5), (S[1], 0.5)]

# R(W,S,W)=0.1 & R(W,S,C)=1
R[(S[1], A[1], S[0])] = 2
R[(S[1], A[1], S[1])] = 2

#############################
######## Functions ##########
#############################


def policyEvaluation(state):
    "Policy Evaluation of S"
    policy = policies[state]
    value = 0
    for nextState in T[(state, policy)]:
        prob = nextState[1]
        reward = R[state, policy, nextState[0]]
        oldValue = oldValues[nextState[0]]
        value += prob*(reward+gamma*oldValue)
    newValues[state] = value


def PolicyImprovement(state):
    "Policy Improvement for S"
    action_s = actions[state]  # Get all posible actions
    maxQVal = float("-inf")
    bestPolicy = ""

    # Iterate over all actions
    for action in action_s:
        value = 0
        # Calculate Q(s,a)
        for nextState in T[(state, action)]:
            prob = nextState[1]
            reward = R[state, action, nextState[0]]
            oldValue = oldValues[nextState[0]]
            value += prob*(reward+gamma*oldValue)

        # Check max value for best policy
        if value > maxQVal:
            maxQVal = value
            bestPolicy = action
    policies[s] = bestPolicy  # Update Best Policy


def checkConvergence():
    "Return if difference between iterations is less than delta"
    for s in S:
        diff = abs(oldValues[s]-newValues[s])
        convergence[s] = diff < delta and (diff > 0 or s is S[2])


def stopIteration():
    "Function for stoping the iteration, if all are true"
    x = True
    for s in S:
        x = x and convergence[s]
    return x


k = 1
while not stopIteration():
    newValues = {x: 0 for x in S}  # Empty for calculation of new values
    for s in S:
        if s is not S[2]:
            policyEvaluation(s)
            PolicyImprovement(s)
    k += 1
    checkConvergence()
    oldValues = newValues  # Update values
    print(f"{bcolors.WARNING}Policies at iteration {k}: {policies}{bcolors.RESET}")
    print(f"{bcolors.WARNING}Values at iteration {k}: {oldValues}{bcolors.RESET}")
    print("----------------------------------------")
    if(k == 1000):
        print(f"{bcolors.FAIL}Iteration will be break{bcolors.RESET}")
        break

print(f"{bcolors.OK}Final Policies: {policies}{bcolors.RESET}")
print(f"{bcolors.OK}Final Values: {oldValues}{bcolors.RESET}")
print(f"{bcolors.OK}Convergence in iteration {k}{bcolors.RESET}")
