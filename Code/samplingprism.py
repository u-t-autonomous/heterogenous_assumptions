from __future__ import division
import stormpy
import stormpy.core
import stormpy.logic
import stormpy.pars
import re
import stormpy.examples
import stormpy.examples.files
import time
import numpy as np
import time
from tqdm import tqdm

from gurobipy import *


# Approximation of binomial cdf with continuity correction for large n
# n: trials, p: success prob, m: starting successes
def BCDF(p, n, m):
    return 1-CDF((m-0.5-(n*p))/math.sqrt(n*p*(1-p)))
def CDF(x):
    return (1.0 + math.erf(x/math.sqrt(2.0)))/2.0


#loading model and specs
def loader():
    path = "one_car.prism"
    # path = "brp_rewards4.pm"
    prism_program = stormpy.parse_prism_program(path)
    print("Building model from {}".format(path))
    formula_str = "P=? [ F ((s=N-1)&(s_p=1)) ]"
    # formula_str = "R=? [ F ((s=5) | (s=0&srep=3)) ]"
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    model = stormpy.build_parametric_model(prism_program, properties)

    print("Model supports parameters: {}".format(model.supports_parameters))
    parameters = model.collect_probability_parameters()
    numstate=model.nr_states
    print("Number of states before bisim:",numstate)
    #    #assert len(parameters) == 2
    print ("Number of params before bisim:",len(parameters))
    #   print(model.model_type)
    # instantiator = stormpy.pars.PDtmcInstantiator(model)
    #print (model.initial_states)
    #gathering parameters in the bisimulated mpdel
    model = stormpy.perform_bisimulation(model, properties, stormpy.BisimulationType.STRONG)
    parameters= model.collect_probability_parameters()
    # parameters_rew = model.collect_reward_parameters()
    # parameters.update(parameters_rew)


    numstate=model.nr_states
    print("Number of states after bisim:",numstate)
    #    #assert len(parameters) == 2
    print ("Number of params after bisim:",len(parameters))
    return parameters,model,properties

# paraminit = [[0 for state in range(numstate)] for state in range(numstate)]
def run_sample(numiter,numsample,thres,direction,parameters,model,properties):
    '''

    :param numiter: number of trials to compute the number of samples that satisfies the probability
    :param numsample: number of sampled pMDPs/pMCs to check
    :param thres: specification threshold
    :param direction: if True, then the spec is \geq, if False, then it is \leq
    :return:
    '''
    #storing the approximate satisfaction probability for each iteration
    counterarray= [0 for _ in range(numiter)]

    instantiator = stormpy.pars.PDtmcInstantiator(model)

    start3 = time.time()
    #for each iteration
    for iter in tqdm(range(numiter)):
        counter=0
        # for each sample
        for i in range(int(numsample)):
            #sample parameters according to the region defined by
            # "Parameter synthesis for Markov models: Faster than ever" paper
            point=dict()
            for x in parameters:

                s = np.random.uniform(1e-5, 1-1e-5)

                point[x] = stormpy.RationalRF(s)
            #check result
            rational_parameter_assignments = dict(
                [[x, stormpy.RationalRF(val)] for x, val in point.items()])
            instantiated_model = instantiator.instantiate(rational_parameter_assignments)
            result = stormpy.model_checking(instantiated_model, properties[0]).at(instantiated_model.initial_states[0])
            #append the counter according to the spec
            if direction==True:
                if float(result)>thres:
                    counter=counter+1
            else:
                if float(result)<thres:
                    counter=counter+1

        counterarray[iter]=counter/((numsample)*1.0)

        if iter>1:
            #print(counterarray[0:iter])
            print("Average violation so far:",np.mean(counterarray[0:iter]))

    #print(dict1)

    t3 = time.time()
    print("Solver time :" + str(t3 - start3))
    return counterarray

def compute_avg_satprob(counterarray,N,eps,flag):
    '''
    :param counterarray: approximate satisfaction probs
    :param N: number of samples
    :param eps:
    :param direction: if True, then the spec is \geq, if False, then it is \leq

    :return:
    '''
    #storing probabilities for each iteration
    valuearray = np.zeros(len(counterarray))

    for iters in range(len(counterarray)):
        print(counterarray[iters])

        val = 0
        val2 = 0
        #compute constraints to remove in the LP
        if flag:
            removeconstraints = int(N * (counterarray[iters]))
        else:
            removeconstraints = int(N * (1 - counterarray[iters]))
        # print(N,removeconstraints,eps)

        val3 = BCDF(eps, N, removeconstraints)

        valuearray[iters] = val3

    print("probability of violating the spec less then the threshold for each iter:")
    print(valuearray)
    print("average value of the array:")
    print(np.mean(valuearray))



parameters,model,properties=loader()
print("1")
numiter=5
numsample=5000
threshold=0.2
direction=False
counter_array=run_sample(numiter,numsample,threshold,direction,parameters,model,properties)
print(counter_array)
#this is violation probability
eps=0.4
compute_avg_satprob(counter_array,numsample,eps,direction)