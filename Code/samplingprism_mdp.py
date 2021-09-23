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
import matplotlib.pyplot as plt
import math

# from gurobipy import *


# Approximation of binomial cdf with continuity correction for large n
# n: trials, p: success prob, m: starting successes
def BCDF(p, n, m):
    return 1-CDF((m-0.5-(n*p))/math.sqrt(n*p*(1-p)))
def CDF(x):
    return (1.0 + math.erf(x/math.sqrt(2.0)))/2.0


#loading model and specs
def loader():
    path = "two_car_mdp.prism"
    prism_program = stormpy.parse_prism_program(path)
    print("Building model from {}".format(path))
    formula_str = 'Pmax=? [!"Crash" U "Goal"]'
    # formula_str = "R=? [ F ((s=5) | (s=0&srep=3)) ]"
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    model = stormpy.build_parametric_model(prism_program)

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
    # model = stormpy.perform_bisimulation(model, properties, stormpy.BisimulationType.STRONG)
    # parameters= model.collect_probability_parameters()
    # parameters_rew = model.collect_reward_parameters()
    # parameters.update(parameters_rew)


    numstate=model.nr_states
    print("Number of states after bisim:",numstate)
    # #    #assert len(parameters) == 2
    # print ("Number of params after bisim:",len(parameters))
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
    # path = "one_car_mdp.prism"
    # prism_program = stormpy.parse_prism_program(path)
    # formula_str = 'Pmax=? [!"Crash" U "Goal"]'
    # prop = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    # model_one_car = stormpy.build_parametric_model(prism_program)
    # parameters_oc = model_one_car.collect_probability_parameters()
    # parameter_oc = parameters_oc.pop()
    # instant_model = stormpy.pars.PMdpInstantiator(model_one_car)




    #storing the approximate satisfaction probability for each iteration
    move_ids = [m_s.id for m_s in model.states for s_i in m_s.labels if 'go' in s_i or 'stop' in s_i]
    action_points = set(range(model.nr_states)) - set(move_ids)
    actions1 = ['go1','stop1']
    actions2 = ['go2','stop2']

    second_car_points = []
    for m_s in model.states:
        flag1 = False
        flag2 = True
        for s_i in m_s.labels:
            if 'go1' in s_i or 'stop1' in s_i:
                flag1 = True
            if 'go2' in s_i or 'stop2' in s_i:
                flag2 = False
        if flag1 and flag2:
            second_car_points.append(m_s.id)
    counterarray= [0 for _ in range(numiter)]

    instantiator = stormpy.pars.PMdpInstantiator(model)

    start3 = time.time()
    #for each iteration
    data_out = dict()
    for iter in range(numiter):
        counter=0
        # for each sample
        for i in tqdm(range(int(numsample))):
            #sample parameters according to the region defined by
            # "Parameter synthesis for Markov models: Faster than ever" paper
            point=dict()
            point2=dict()
            Agent1_pol = dict()
            Agent2_pol = dict()
            for x in parameters:

                s = np.random.uniform(1e-5, 1-1e-5)
                s2 = np.random.uniform(1e-5,1-1e-5)
                point[x] = stormpy.RationalRF(s)
                point2[x] = stormpy.RationalRF(s2)
            #check result
            rational_parameter_assignments = dict(
                [[x, stormpy.RationalRF(val)] for x, val in point.items()])
            rational_parameter_assignments2 = dict(
                [[x, stormpy.RationalRF(val)] for x, val in point2.items()])
            instantiated_model = instantiator.instantiate(rational_parameter_assignments)
            instantiated_model2 = instantiator.instantiate(rational_parameter_assignments2)
            time_t1 = time.time()
            result = stormpy.model_checking(instantiated_model, properties[0],extract_scheduler=True)
            result2 = stormpy.model_checking(instantiated_model2, properties[0],extract_scheduler=True)
            print('Synth RT {}'.format(time.time()-time_t1))

            # rational_parameters_oc = dict([[parameter_oc, stormpy.RationalRF(s)]])
            # ins_oc = instant_model.instantiate(rational_parameters_oc)
            # result_oc = stormpy.model_checking(ins_oc, prop[0], extract_scheduler=True)


            for s_i in action_points:
                for l_i in model.states[s_i].labels:
                    if 's' in l_i and not 'Crash' in l_i:
                        s_state = l_i
                    elif 'x' in l_i:
                        x_state = l_i
                    elif 'p' in l_i:
                        p_state = l_i
                hold_state = model.states[int(re.findall('\\d+', str(model.states[s_i].actions[result.scheduler.get_choice(s_i).get_deterministic_choice()].transitions))[0])]
                next_action = result.scheduler.get_choice(hold_state).get_deterministic_choice()
                next_state = model.states[int(re.findall('\\d+', str(hold_state.actions[int(next_action)].transitions))[0])]
                if 'Crash' not in next_state.labels and 'Goal' not in next_state.labels:
                    act_tup = tuple()
                    act_tup += ([l_ind for l_ind,l_a in enumerate(actions1) if l_a in next_state.labels][0],)
                    act_tup += ([l_ind for l_ind,l_a in enumerate(actions2) if l_a in next_state.labels][0],)
                    Agent1_pol.update({(s_state, x_state, p_state): act_tup[0]})
                hold_state2 = model.states[int(re.findall('\\d+', str(model.states[s_i].actions[result2.scheduler.get_choice(s_i).get_deterministic_choice()].transitions))[0])]
                next_action2 = result2.scheduler.get_choice(hold_state2).get_deterministic_choice()
                next_state2 = model.states[int(re.findall('\\d+', str(hold_state2.actions[int(next_action2)].transitions))[0])]
                if 'Crash' not in next_state2.labels and 'Goal' not in next_state2.labels:
                    act_tup2 = tuple()
                    act_tup2 += ([l_ind for l_ind,l_a in enumerate(actions1) if l_a in next_state2.labels][0],)
                    act_tup2 += ([l_ind for l_ind,l_a in enumerate(actions2) if l_a in next_state2.labels][0],)
                    Agent2_pol.update({(s_state,x_state,p_state):act_tup2[1]})

            # for s_i in range(model_one_car.nr_states):
            #     for l_i in model_one_car.states[s_i].labels:
            #         if 's' in l_i and not 'Crash' in l_i:
            #             s_state = l_i
            #         elif 'x' in l_i:
            #             x_state = l_i
            #         elif 'p' in l_i:
            #             p_state = l_i
            #     for x_state in range(6):
            #         Agent1_pol.update({(s_state,'x{}'.format(x_state),p_state):result_oc.scheduler.get_choice(s_i).get_deterministic_choice()%2})

            # for s_i in second_car_points:
            #     for l_i in model.states[s_i].labels:
            #         if 's' in l_i and not 'Crash' in l_i and not 'stop' in l_i:
            #             s_state = l_i
            #         elif 'x' in l_i:
            #             x_state = l_i
            #         elif 'p' in l_i and not 'stop' in l_i:
            #             p_state = l_i
            #     Agent2_pol.update({(s_state,x_state,p_state):result2.scheduler.get_choice(s_i).get_deterministic_choice()})

            # result = stormpy.model_checking(instantiated_model, properties[0]).at(instantiated_model.initial_states[0])
            #append the counter according to the spec
            tc_dtmc = "two_car_dtmc.prism"
            with open("two_car_mdp_template.txt") as f:
                with open(tc_dtmc, "w+") as f1:
                    for line in f:
                        f1.write(line)

            dtmc_file = open(tc_dtmc,"a+")
            dtmc_file.write("module car1policy\n\tc1_go1 : bool init false;\n\tc1_stop1 : bool init false;\n\n\t[go] c1_go1 -> 1:(c1_go1'=false);\n\t[stop] c1_stop1 -> 1:(c1_stop1'=false);\n")

            pol1_lead = "\t[assign] !carpol1 &"
            out1_ind = ["(c1_go1'=true);\n","(c1_stop1'=true);\n"]
            for s_z,x,p in Agent1_pol:
                s_num = int(s_z[1:])
                x_num = int(x[1:])
                p_num = int(p[1:])

                state_in = "(s={}) & (s2={}) & (s_p={}) -> ".format(s_num,x_num,p_num)
                pol_ind = 0 if Agent1_pol[(s_z,x,p)] == int(0) else 1
                dtmc_file.write(pol1_lead+state_in+out1_ind[pol_ind])

            dtmc_file.write('endmodule\n')

            dtmc_file.write("module car2policy\n\tc2_go2 : bool init false;\n\tc2_stop2 : bool init false;\n\n\t[go2] c2_go2 -> 1:(c2_go2'=false);\n\t[stop2] c2_stop2 -> 1:(c2_stop2'=false);\n")
            pol2_lead = "\t[assign] !carpol2 &"
            out2_ind = ["(c2_go2'=true);\n", "(c2_stop2'=true);\n"]
            for s_z, x, p in Agent2_pol:
                s_num = int(s_z[1:])
                x_num = int(x[1:])
                p_num = int(p[1:])

                state_in = "(s={}) & (s2={}) & (s_p={}) -> ".format(s_num, x_num, p_num)
                pol_ind = 0 if Agent2_pol[(s_z, x, p)] == int(0) else 1
                dtmc_file.write(pol2_lead + state_in + out2_ind[pol_ind])

            dtmc_file.write('endmodule\n')
            dtmc_file.write('\nlabel "Crash" = (crash=true);\nlabel "Goal" = (s=N)|(s2=N);\n')
            dtmc_file.close()

            prism_program = stormpy.parse_prism_program(tc_dtmc)
            dtmc_model = stormpy.build_parametric_model(prism_program)
            dtmc_params = dtmc_model.collect_probability_parameters()

            time_t2 = time.time()
            sT = 0.75 # np.random.uniform(1e-5, 1 - 1e-5)
            rational_parameter_assignments = dict([[dtmc_params.pop(), stormpy.RationalRF(sT)] ])
            combined_instantiator = stormpy.pars.PDtmcInstantiator(dtmc_model)
            instantiated_dtmc = combined_instantiator.instantiate(rational_parameter_assignments)
            print(instantiated_dtmc.nr_states)
            result_dtmc = stormpy.model_checking(instantiated_dtmc, properties[0]).at(instantiated_dtmc.initial_states[0])
            data_out.update({(s,s2):result_dtmc})
            print('Verification Time {}'.format(time.time()-time_t2))

            if direction==True:
                if float(result.at(instantiated_model.initial_states[0]))>thres:
                    counter=counter+1
            else:
                if float(result.at(instantiated_model.initial_states[0]))<thres:
                    counter=counter+1

        counterarray[iter]=counter/((numsample)*1.0)

        if iter>1:
            #print(counterarray[0:iter])
            print("Average violation so far:",np.mean(counterarray[0:iter]))

    #print(dict1)

    t3 = time.time()
    print("Solver time :" + str(t3 - start3))
    return counterarray,data_out

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
numiter=2
numsample=1
threshold=0.75
direction=True
counter_array,data_out=run_sample(numiter,numsample,threshold,direction,parameters,model,properties)
fig = plt.figure()
ax = plt.axes(projection='3d')
xline = []
yline = []
zline = []
for x,y in data_out:
    xline.append(x)
    yline.append(y)
    zline.append(data_out[(x,y)])

a_li = np.asarray([xline,yline,zline])
# np.savetxt('TwoCarThreshold075.csv',a_li.T,delimiter=',')
ax.scatter3D(xline,yline,zline,c=zline)
ax.set_xlabel(r'$p_1$')
ax.set_ylabel(r'$p_2$')
ax.set_zlabel(r'$p_T$')
plt.show()
print(counter_array)
#this is violation probability
eps=0.1
compute_avg_satprob(counter_array,numsample,eps,direction)