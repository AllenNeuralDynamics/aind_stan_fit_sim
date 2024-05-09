# %%
import nest_asyncio
import stan
import numpy as np
import pandas as pd
from os import path
from RLmodels import QLearningModel
# from RLmodels import qLearningModel_5params_simNoPlot
from RLmodels import RestlessBanditDecoupled
from RLmodels import QLearningModelSim
import matplotlib.pyplot as plt
import seaborn as sns
nest_asyncio.apply()

# %%
# params = [0.2, 0.8, 0.2, 5, 0.2] # aN, aP, aF, beta, bias
# choice = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
# outcome = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
# model = QLearningModel(params)
# model.fit(choice, outcome)
# # model.plot_values()
# np.exp(model.LH/len(model.choice))

# %%
# maxTrial = 500
# rlSim = QLearningModelSim(params, maxTrial = maxTrial, randomSeed=5)
# rlSim.run_simulation()  
# rlSim.plotSim()

# %%
# simulation parameters
paramsAni = [0.2, 0.8, 0.2, 5, 0.2]
paramNames = ['aN', 'aP', 'aF', 'beta', 'bias']
sessNum = 10
maxTrial = 500
paramsSim = np.zeros((sessNum, len(paramsAni)-1))

total = 20
aN = np.random.beta(total*paramsAni[0], total*(1-paramsAni[0]), size=sessNum)
aP = np.random.beta(total*paramsAni[1], total*(1-paramsAni[1]), size=sessNum)
aF = np.random.beta(total*paramsAni[2], total*(1-paramsAni[2]), size=sessNum)
beta = np.random.beta(total * 0.5, total * 0.5, size=sessNum) + 4.5
bias = np.random.beta(total*paramsAni[4], total*(1-paramsAni[4]), size=sessNum)
sessionLen = np.random.random_integers(300, 500, sessNum)

paramsSim = pd.DataFrame({paramNames[0]:aN,
                        paramNames[1]:aP,
                        paramNames[2]:aF,
                        paramNames[3]:beta,
                        paramNames[4]:bias,
                        'sessionLen': sessionLen
                        })

sns.pairplot(paramsSim, diag_kws={'bins': 10})

# %%
# session simulation
choiceSim = []
outcomeSim = []
sessionLen = []


for sessionInd in range(sessNum):
    paramsCurr = paramsSim.loc[sessionInd].values
    paramsCurr = paramsCurr[:-1]
    rlSim = QLearningModelSim(paramsCurr, maxTrial = paramsSim.loc[sessionInd, 'sessionLen'], randomSeed=sessionInd)
    rlSim.run_simulation() 
    choiceSim.append(rlSim.allChoices) 
    outcomeSim.append(rlSim.allRewards) 
    sessionLen.append(len(rlSim.allChoices))

# %%
plt.hist(sessionLen)

# %%
# make data for stan
maxLen = max(sessionLen)
sessionNum = sessNum
allChoiceArray = choiceSim
allOutcomeArray = outcomeSim
allChoiceArray = np.array([np.pad(choiceSimCurr, (0, maxLen-sessionLenCurr), mode='constant') for choiceSimCurr, sessionLenCurr in zip(choiceSim, sessionLen)]).astype(int)
allOutcomeArray = np.array([np.pad(outcomeSimCurr, (0, maxLen-sessionLenCurr), mode='constant') for outcomeSimCurr, sessionLenCurr in zip(outcomeSim, sessionLen)]).astype(int)
sim_data = {"N": sessionNum,
            "T": maxLen,
            "Tsesh": sessionLen,
            "choice": allChoiceArray,
            "outcome": allChoiceArray}

# %%
# Read the Stan model from a file
model = '/code/stan_qLearning_5params.stan'
with open(model, "r") as file:
    model_code = file.read()

# %%
posterior = stan.build(model_code, data=sim_data)
fit = posterior.sample(num_chains=16, num_samples=500, num_warmup=200)




