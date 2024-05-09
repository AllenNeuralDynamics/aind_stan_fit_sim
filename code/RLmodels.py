import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import logistic
from collections import namedtuple
from scipy.stats import pearsonr

class QLearningModel:
    def __init__(self, start_values):
        self.alpha_npe = start_values[0]
        self.alpha_ppe = start_values[1]
        self.alpha_forget = start_values[2]
        self.beta = start_values[3]
        self.bias = start_values[4]

        # Initialize attributes
        self.Q = None
        self.pe = None
        self.choice = None
        self.outcome = None
        self.pChoice = None
        self.LH = None

    def fit(self, choice, outcome):
        trials = len(choice)
        self.choice = choice
        self.outcome = outcome
        self.Q = np.zeros((trials, 2))
        self.pe = np.zeros(trials)

        for t in range(trials - 1):
            if choice[t] == 1:  # right choice
                self.Q[t + 1, 0] = self.alpha_forget * self.Q[t, 0]
                self.pe[t] = outcome[t] - self.Q[t, 1]
                if self.pe[t] < 0:
                    self.Q[t + 1, 1] = self.Q[t, 1] + self.alpha_npe * self.pe[t]
                else:
                    self.Q[t + 1, 1] = self.Q[t, 1] + self.alpha_ppe * self.pe[t]
            else:  # left choice
                self.Q[t + 1, 1] = self.alpha_forget * self.Q[t, 1]
                self.pe[t] = outcome[t] - self.Q[t, 0]
                if self.pe[t] < 0:
                    self.Q[t + 1, 0] = self.Q[t, 0] + self.alpha_npe * self.pe[t]
                else:
                    self.Q[t + 1, 0] = self.Q[t, 0] + self.alpha_ppe * self.pe[t]

        # Softmax rule
        prob_choice = 1 / (1 + np.exp(-self.beta * (self.Q[:, 1] - self.Q[:, 0]) + self.bias))

        if choice[-1] == 1:
            self.pe[-1] = self.outcome[-1] - self.Q[-1, 1]
        else:
            self.pe[-1] = self.outcome[-1] - self.Q[-1, 0]
        
        prob_chosen = prob_choice.copy()
        prob_chosen[choice == 0] = 1 - prob_choice[choice == 0]
        self.pChoice = prob_chosen
        LH = self._likelihood(choice, prob_choice)
        self.LH = LH

        return self

    def _likelihood(self, choice, prob_choice):
        return np.log(np.prod(prob_choice[choice == 1]) * np.prod(1 - prob_choice[choice == 0]))

    def plot_values(self):

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.Q[:, 0], label='Q-value (Left)')
        plt.plot(self.Q[:, 1], label='Q-value (Right)')
        plt.xlabel('Trial')
        plt.ylabel('Q-value')
        plt.title('Q-values over Trials')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.pe, color='red')
        plt.xlabel('Trial')
        plt.ylabel('Prediction Error')
        plt.title('Prediction Error over Trials')

        plt.tight_layout()
        plt.show()

# # Example usage:
# if __name__ == "__main__":
#     start_values = [0.1, 0.2, 0.3, 0.4, 0.5]
#     choice = np.array([1, 0, 1, 1, 0])  # 1 for right choice, 0 for left choice
#     outcome = np.array([1, 0, 1, 1, 0])  # Example outcome data

#     model = QLearningModel(start_values)
#     model.fit(choice, outcome)
#     model.plot_values()

# simulation

# define decoupled task
class RestlessBanditDecoupled:
    def __init__(self, rewardProbabilities=None, blockLength=[20, 35], maxTrials=1000, RandomSeed=1):
        self.RewardProbabilitiesList = rewardProbabilities if rewardProbabilities is not None else [90, 50, 10]
        self.RewardProbabilities = np.random.choice(self.RewardProbabilitiesList, size=2, replace=True)
        while np.all(self.RewardProbabilities == 10):
            self.RewardProbabilities = np.random.choice(self.RewardProbabilitiesList, size=2, replace=True)

        self.BlockProbs = self.RewardProbabilities
        self.BlockLength = blockLength
        self.MaxTrials = maxTrials
        self.RandomSeed = RandomSeed
        self.AllRewards = np.full((self.MaxTrials, 2), np.nan)
        self.AllChoices = np.full((self.MaxTrials, 2), np.nan)
        self.BlockSwitch_Flag = True
        self.NewBlockL_Flag = False
        self.NewBlockR_Flag = False
        self.BlockSwitchL = [1]
        self.BlockSwitchR = [1]
        self.Trial = 0
        self.BlockEndL = np.random.randint(min(self.BlockLength), max(self.BlockLength)) + self.Trial
        self.BlockEndR = np.random.randint(min(self.BlockLength), max(self.BlockLength)) + self.Trial
        self.BlockSwitchL.append(self.BlockEndL)
        self.BlockSwitchR.append(self.BlockEndR)
        self.BlockStagger = int(np.ceil(np.mean(self.BlockLength) / 2))
        self.HigherCount = [1, 0] if self.RewardProbabilities[0] > self.RewardProbabilities[1] else [0, 1]
        self.PersevCount = [0, 0]
        self.highRight = np.array([])
        self.highLeft = np.array([])
        self.PersevLeft = np.array([])
        self.PersevRight = np.array([])
        self.HighCountBi = np.append(self.HigherCount, self.Trial)
        self.ITI = 0
        if  np.random.binomial(1, 0.5) == 0:
            self.BlockEndL = self.BlockEndL - self.BlockStagger
            self.BlockSwitchL[-1] = self.BlockEndL
        else:
            self.BlockEndR = self.BlockEndR - self.BlockStagger
            self.BlockSwitchR[-1] = self.BlockEndR
        np.random.seed(self.RandomSeed)
    def inputChoice(self, currChoice):
        self.Trial += 1
        if self.BlockSwitch_Flag and self.Trial > 1:
            self.BlockSwitch_Flag = False
        # if self.Trial == 235:
        #     print('Target')
        if currChoice == [1, 0] and self.RewardProbabilities[0] == 10:
            self.PersevCount[0] += 1
        elif currChoice == [0, 1] and self.RewardProbabilities[1] == 10:
            self.PersevCount[1] += 1

        if self.PersevCount[0] >= 4 and self.RewardProbabilities[0] == 10:
            self.PersevLeft = np.append(self.PersevLeft, self.Trial-1)
            self.BlockEndL += self.PersevCount[0]
            self.BlockSwitchL[-1] = self.BlockEndL
            self.BlockEndR += self.PersevCount[0]
            self.BlockSwitchR[-1] = self.BlockEndR
            self.PersevCount[0] = 0
            self.NewBlockL_Flag = False
            self.NewBlockR_Flag = False
        elif self.PersevCount[1] >= 4 and self.RewardProbabilities[1] == 10:
            self.PersevRight = np.append(self.PersevRight, self.Trial-1)
            self.BlockEndL += self.PersevCount[1]
            self.BlockSwitchR[-1] = self.BlockEndL
            self.BlockEndR += self.PersevCount[1]
            self.BlockSwitchR[-1] = self.BlockEndL
            self.PersevCount[1] = 0
            self.NewBlockL_Flag = False
            self.NewBlockR_Flag = False

        if self.NewBlockL_Flag:
            self.generateBlockL()

        if self.NewBlockR_Flag:
            self.generateBlockR()

        if currChoice == [1, 0]:
            self.AllChoices[self.Trial - 1] = [1, 0]
            if self.RewardProbabilities[0] > np.random.randint(0, 100):
                self.AllRewards[self.Trial - 1] = [1, 0]
            else:
                self.AllRewards[self.Trial - 1] = [0, 0]
        elif currChoice == [0, 1]:
            self.AllChoices[self.Trial - 1] = [0, 1]
            if self.RewardProbabilities[1] > np.random.randint(0, 100):
                self.AllRewards[self.Trial - 1] = [0, 1]
            else:
                self.AllRewards[self.Trial - 1] = [0, 0]
        else:
            raise ValueError('Choice should be [1 0] (left choice) or [0 1] (right choice)')

        if self.Trial == self.BlockSwitchL[-1]:
            self.NewBlockL_Flag = True

        if self.Trial == self.BlockSwitchR[-1]:
            self.NewBlockR_Flag = True

        self.ITI = -np.log(np.random.rand()) / 0.3

    def generateBlockL(self):
        self.NewBlockL_Flag = False
        self.BlockSwitch_Flag = True
        self.BlockEndL =  np.random.randint(np.min(self.BlockLength), np.max(self.BlockLength)) + self.Trial
        self.BlockSwitchL.append(self.BlockEndL)
        if self.HigherCount[0] >= 3:
            self.highLeft = np.append(self.highLeft, self.Trial)
            self.RewardProbabilities[0] = 10
            self.HigherCount[0] = 0
            self.HigherCount[1] += 1
        else:
            tmp = self.RewardProbabilities[0]
            while self.RewardProbabilities[0] == tmp:
                self.RewardProbabilities[0] = np.random.choice(self.RewardProbabilitiesList)
            if self.RewardProbabilities[0] > self.RewardProbabilities[1]:
                self.HigherCount[0] += 1
                self.HigherCount[1] = 0
            elif self.RewardProbabilities[0] < self.RewardProbabilities[1]:
                self.HigherCount[1] += 1
                self.HigherCount[0] = 0
            else:
                self.HigherCount[0] += 1
                self.HigherCount[1] += 1

        if np.all(self.RewardProbabilities == 10):
            self.BlockEndL -= self.BlockStagger
            self.BlockSwitchL[-1] = self.BlockEndL
            self.HigherCount[0] = 0
            self.HigherCount[1] = 0
            self.BlockEndR = self.Trial - 1
            self.BlockSwitchR[-1] = self.BlockEndR
            self.generateBlockR()
        elif not self.NewBlockR_Flag:
            self.BlockProbs = np.vstack([self.BlockProbs, self.RewardProbabilities])
        self.HighCountBi = np.vstack((self.HighCountBi, np.append(self.HigherCount, self.Trial)))

    def generateBlockR(self):
        self.NewBlockR_Flag = False 
        self.BlockSwitch_Flag = True
        self.BlockEndR =  np.random.randint(np.min(self.BlockLength), np.max(self.BlockLength)) + self.Trial
        self.BlockSwitchR.append(self.BlockEndR)
        if self.HigherCount[1] >= 3:
            self.highRight = np.append(self.highRight, self.Trial)
            self.RewardProbabilities[1] = 10
            self.HigherCount[1] = 0
            self.HigherCount[0] += 1
        else:
            tmp = self.RewardProbabilities[1]
            while self.RewardProbabilities[1] == tmp:
                self.RewardProbabilities[1] = np.random.choice(self.RewardProbabilitiesList)
            if self.RewardProbabilities[1] > self.RewardProbabilities[0]:
                self.HigherCount[1] += 1
                self.HigherCount[0] = 0
            elif self.RewardProbabilities[1] < self.RewardProbabilities[0]:
                self.HigherCount[0] += 1
                self.HigherCount[1] = 0
            else:
                self.HigherCount[0] += 1
                self.HigherCount[1] += 1

        if np.all(self.RewardProbabilities == 10):
            self.BlockEndR -= self.BlockStagger
            self.BlockSwitchR[-1] = self.BlockEndR
            self.HigherCount[0] = 0
            self.HigherCount[1] = 0
            self.BlockEndL = self.Trial - 1
            self.BlockSwitchL[-1] = self.BlockEndL
            self.generateBlockL()
        elif not self.NewBlockR_Flag:
            self.BlockProbs = np.vstack([self.BlockProbs, self.RewardProbabilities])

        self.HighCountBi = np.vstack((self.HighCountBi, np.append(self.HigherCount, self.Trial)))

def softmax(x):
    y = 1/(1 + np.exp(-x))
    return y

# define simulator as class

class QLearningModelSim:
    def __init__(self, params, maxTrial, randomSeed=1, **kwargs):
       
        self.params = params
        self.maxTrial = maxTrial
        self.randomSeed = randomSeed
        self.taskType = kwargs.get('taskType', 'decoupled')
        self.blockLength = kwargs.get('blockLength', [20, 35])
        self.rwdProbs = kwargs.get('rwdProbs', [90, 50, 10])
        self.ITIparam = kwargs.get('ITIparam', 0.3)
        self.Q = np.zeros((maxTrial + 1, 2))
        self.probChoice = np.full((maxTrial), np.nan)
        self.pe = np.full((maxTrial), np.nan)
        self.allRewards = np.full((maxTrial), np.nan)
        self.allChoices = np.full((maxTrial), np.nan)
        self.blockProbs = []
        self.blockSwitch = []
        self.highRight = []
        self.highLeft = []
        self.PersevRight = []
        self.PersevLeft = []
        self.HighCountBi = []
        np.random.seed(self.randomSeed)

    def run_simulation(self):
        alphaNPE, alphaPPE, alphaForget, beta, bias = self.params

        if self.taskType == 'coupled':
            p = RestlessBandit(randomSeed=self.randomSeed, blockLength=self.blockLength, maxTrials=self.maxTrial, rewardProbabilities=self.rwdProbs)
        elif self.taskType == 'switch':
            p = RestlessBanditSwitch(randomSeed=self.randomSeed, blockLength=self.blockLength, maxTrials=self.maxTrial)
        elif self.taskType == 'decoupled':
            p = RestlessBanditDecoupled(RandomSeed=self.randomSeed, blockLength=self.blockLength, maxTrials=self.maxTrial, rewardProbabilities=self.rwdProbs)

        for currT in range(self.maxTrial):
            pRight = softmax((beta * (self.Q[currT, 1] - self.Q[currT, 0]) + bias))
            if np.random.binomial(1, pRight) == 0:
                self.probChoice[currT] = 1 - pRight
                p.inputChoice([1, 0])
                self.allChoices[currT] = 0
                self.allRewards[currT] = p.AllRewards[currT, 0] 
                rpe = p.AllRewards[currT, 0] - self.Q[currT, 0]
                self.pe[currT] = rpe
                if rpe >= 0:
                    self.Q[currT + 1, 0] = self.Q[currT, 0] + alphaPPE * rpe
                else:
                    self.Q[currT + 1, 0] = self.Q[currT, 0] + alphaNPE * rpe
                self.Q[currT + 1, 1] = self.Q[currT, 1] * alphaForget
            else:
                self.probChoice[currT] = pRight
                p.inputChoice([0, 1])
                self.allChoices[currT] = 1
                self.allRewards[currT] = p.AllRewards[currT, 1]
                rpe = p.AllRewards[currT, 1] - self.Q[currT, 1]
                self.pe[currT] = rpe
                if rpe >= 0:
                    self.Q[currT + 1, 1] = self.Q[currT, 1] + alphaPPE * rpe
                else:
                    self.Q[currT + 1, 1] = self.Q[currT, 1] + alphaNPE * rpe
                self.Q[currT + 1, 0] = self.Q[currT, 0] * alphaForget

            if currT == 0:
                self.blockProbs = p.RewardProbabilities
            else:
                self.blockProbs = np.vstack([self.blockProbs, p.RewardProbabilities])

        self.highRight = p.highRight
        self.highLeft = p.highLeft
        self.PersevLeft = p.PersevLeft
        self.PersevRight = p.PersevRight
        self.HighCountBi = p.HighCountBi

    def plotSim(self):
        plt.figure(figsize=(20, 9))
        plt.subplot(4, 1, 1)
        x = np.tile(np.array(range(self.maxTrial)), (2, 1))
        y = np.concatenate((np.zeros((1,self.maxTrial)), (self.allChoices[:, np.newaxis].T-0.5) * (self.allRewards[:, np.newaxis].T + 1)), axis=0)
        plt.plot(np.array(range(self.maxTrial)), self.Q[:-1,1] - self.Q[:-1,0], color = [0.7, 0.7, 0.7])
        plt.plot(x, y, c = 'k', lw = 0.5);
        plt.scatter(self.PersevLeft, -np.ones_like(self.PersevLeft), c = 'm', s = 5, label = 'L')
        plt.scatter(self.PersevRight, np.ones_like(self.PersevRight), c = 'c', s = 5, label = 'R')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(np.array(range(self.maxTrial)), self.Q[:-1,0], c = 'm' , label = 'L')
        plt.plot(np.array(range(self.maxTrial)), self.Q[:-1,1], c = 'c', label = 'R')

        plt.subplot(4, 1, 3)
        plt.plot(self.blockProbs[:,0], 'm', label = 'L')
        plt.plot(self.blockProbs[:,1], 'c', label = 'R')
        plt.scatter(self.highRight, np.ones_like(self.highRight), c = 'c')
        plt.scatter(self.highLeft, np.ones_like(self.highLeft), c = 'm')

        plt.subplot(4, 1, 4)
        plt.plot(self.HighCountBi[:,2], self.HighCountBi[:,0], c = 'm')
        plt.plot(self.HighCountBi[:,2], self.HighCountBi[:,1], c = 'c')


def myPairPlot(df):
    g = sns.pairplot(paramsSim, corner=True)
    g.map_lower(corrfunc)
    plt.show()


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)