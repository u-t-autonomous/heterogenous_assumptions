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
# import matplotlib.pyplot as plt
from itertools import product

# from gurobipy import *


g_agents = 2
g_size = 3
g_obs_region = [g_size-1,g_size,g_size-1,g_size]
g_boxes = [g_obs_region,[g_size-2,g_size-1,g_size-1,g_size]]


# g_agents = 3
# g_size = 3
# g_obs_region = [g_size-1,g_size,g_size-1,g_size]
# g_boxes = [g_obs_region,[g_size-2,g_size-1,g_size-1,g_size]]
# g_boxes = [g_obs_region,[g_size-2,g_size-1,g_size-1,g_size],[g_size-1,g_size,g_size-2,g_size-1]]

def build_model_grid(agents,size,obs_box,policy=None,run_box=False):
	obs_states = (obs_box[1]-obs_box[0]+1)*(obs_box[3]-obs_box[2]+1)
	act_list = ['up','down','left','right']
	if policy:
		fname = "{}_car_grid_dtmc.pm".format(agents)
	else:
		fname = "{}_car_grid.prism".format(agents)
	mdp_file = open(fname, "w+")
	if policy:
		mdp_file.write('dtmc\n\n')
	else:
		mdp_file.write('mdp\n\n')
	mdp_file.write('const int xSize = {};\n'.format(size))
	mdp_file.write('const int ySize = {};\n'.format(size))
	mdp_file.write('const int obsXmin = {};\n'.format(obs_box[0]))
	mdp_file.write('const int obsYmin = {};\n'.format(obs_box[2]))
	mdp_file.write('const int obsXmax = {};\n'.format(obs_box[1]))
	mdp_file.write('const int obsYmax = {};\n'.format(obs_box[3]))
	if run_box:
		mdp_file.write('const double pL=0.5;\n')
	else:
		mdp_file.write('const double pL;\n')
	T_string = '(xA0 = xSize & yA0 = ySize)'
	Cr_string = '(xA0 = xObs & yA0 = yObs)|(xA0=xA1 & yA0= yA1)'
	act_strings = ['formula action0 = ua0|da0|ra0|la0;\n']
	for i in range(1,agents):
		T_string += '& (xA{} = xSize-{} & yA{} = ySize)'.format(i,i,i)
		Cr_string += '| (xA{} = xObs & yA{} = yObs)'.format(i,i)
		act_strings.append('formula action{} = ua{}|da{}|ra{}|la{};\n'.format(i,i,i,i,i))
		for j in range(i+1,agents):
			Cr_string += '|(xA{} = xA{} & yA{} = yA{})'.format(i,j,i,j)
	mdp_file.write('formula T = '+T_string+';\n')
	mdp_file.write('formula Cr = ' + Cr_string + ';\n')
	for f_i,f_n in enumerate(act_strings):
		mdp_file.write(f_n)
		if policy:
			mdp_file.write('formula carpol{} = c{}_up{}|c{}_down{}|c{}_left{}|c{}_right{};\n'.format(f_i,f_i,f_i,f_i,f_i,f_i,f_i,f_i,f_i))

	for i in range(agents):
		mdp_file.write('module car{}\n'.format(i))
		mdp_file.write('\txA{}: [0..xSize] init {};\n'.format(i,i))
		mdp_file.write('\tyA{}: [0..ySize] init 0;\n'.format(i))
		mdp_file.write('\tua{}: bool init false;\n'.format(i))
		mdp_file.write('\tda{}: bool init false;\n'.format(i))
		mdp_file.write('\tra{}: bool init false;\n'.format(i))
		mdp_file.write('\tla{}: bool init false;\n\n'.format(i))
		if policy:
			mdp_file.write('\tassign_flag{}: bool init true;\n\n'.format(i))
			mdp_file.write("\t[assign] assign_flag{} -> (assign_flag{}'=false);\n".format(i,i))
			mdp_file.write("\t[up{}] (!T & !Cr & !action{} & !assign_flag{}) -> (ua{}'=true);\n".format(i,i,i,i))
			mdp_file.write("\t[down{}] (!T & !Cr & !action{} & !assign_flag{}) -> (da{}'=true);\n".format(i,i,i,i))
			mdp_file.write("\t[left{}] (!T & !Cr & !action{} & !assign_flag{}) -> (la{}'=true);\n".format(i,i,i,i))
			mdp_file.write("\t[right{}] (!T & !Cr & !action{} & !assign_flag{}) -> (ra{}'=true);\n\n".format(i,i,i,i))
			mdp_file.write("\t[move] (yA{} < ySize & ua{}) -> (ua{}'=false) & (yA{}'=yA{}+1) & (assign_flag{}'=true);\n".format(i, i, i, i, i,i))
			mdp_file.write("\t[move] (yA{} = ySize & ua{}) -> (ua{}'=false) & (yA{}'=yA{}) & (assign_flag{}'=true);\n".format(i, i, i, i, i,i))
			mdp_file.write("\t[move] (xA{} < xSize & ra{}) -> (ra{}'=false) & (xA{}'=xA{}+1) & (assign_flag{}'=true);\n".format(i, i, i, i, i,i))
			mdp_file.write("\t[move] (xA{} = xSize & ra{}) -> (ra{}'=false) & (xA{}'=xA{}) & (assign_flag{}'=true);\n".format(i, i, i, i, i,i))
			mdp_file.write("\t[move] (yA{} = 0 & da{}) -> (da{}'=false) & (yA{}'=yA{}) & (assign_flag{}'=true);\n".format(i, i, i, i, i,i))
			mdp_file.write("\t[move] (yA{} > 0 & da{}) -> (da{}'=false) & (yA{}'=yA{}-1) & (assign_flag{}'=true);\n".format(i, i, i, i, i,i))
			mdp_file.write("\t[move] (xA{} = 0 & la{}) -> (la{}'=false) & (xA{}'=xA{}) & (assign_flag{}'=true);\n".format(i, i, i, i, i,i))
			mdp_file.write("\t[move] (xA{} > 0 & la{}) -> (la{}'=false) & (xA{}'=xA{}-1) & (assign_flag{}'=true);\n\n".format(i, i, i, i, i,i))

		else:
			mdp_file.write("\t[up{}] (!T & !Cr & !action{}) -> (ua{}'=true);\n".format(i,i,i))
			mdp_file.write("\t[down{}] (!T & !Cr & !action{}) -> (da{}'=true);\n".format(i,i,i))
			mdp_file.write("\t[left{}] (!T & !Cr & !action{}) -> (la{}'=true);\n".format(i,i,i))
			mdp_file.write("\t[right{}] (!T & !Cr & !action{}) -> (ra{}'=true);\n\n".format(i,i,i))
			mdp_file.write("\t[move] (yA{} < ySize & ua{}) -> (ua{}'=false) & (yA{}'=yA{}+1);\n".format(i,i,i,i,i))
			mdp_file.write("\t[move] (yA{} = ySize & ua{}) -> (ua{}'=false) & (yA{}'=yA{});\n".format(i,i,i,i,i))
			mdp_file.write("\t[move] (xA{} < xSize & ra{}) -> (ra{}'=false) & (xA{}'=xA{}+1);\n".format(i,i,i,i,i))
			mdp_file.write("\t[move] (xA{} = xSize & ra{}) -> (ra{}'=false) & (xA{}'=xA{});\n".format(i,i,i,i,i))
			mdp_file.write("\t[move] (yA{} = 0 & da{}) -> (da{}'=false) & (yA{}'=yA{});\n".format(i,i,i,i,i))
			mdp_file.write("\t[move] (yA{} > 0 & da{}) -> (da{}'=false) & (yA{}'=yA{}-1);\n".format(i,i,i,i,i))
			mdp_file.write("\t[move] (xA{} = 0 & la{}) -> (la{}'=false) & (xA{}'=xA{});\n".format(i,i,i,i,i))
			mdp_file.write("\t[move] (xA{} > 0 & la{}) -> (la{}'=false) & (xA{}'=xA{}-1);\n\n".format(i,i,i,i,i))

		mdp_file.write('\t[doneT] T -> 1:true;\n')
		mdp_file.write('\t[doneC] Cr-> 1:true;\n')
		mdp_file.write('endmodule\n\n')

	mdp_file.write('module obstacle\n')
	mdp_file.write('\txObs: [obsXmin..obsXmax] init obsXmin;\n')
	mdp_file.write('\tyObs: [obsYmin..obsYmax] init obsYmin;\n')
	f_obs = '\t[move] true -> (1-pL):true'
	for i in range(obs_box[0],obs_box[1]+1):
		for j in range(obs_box[2],obs_box[3]+1):
			f_obs += "+ pL/{}:(xObs'={})&(yObs'={})".format(obs_states,i,j)
	f_obs += ';\n'
	mdp_file.write(f_obs)
	mdp_file.write('endmodule\n\n')

	if policy:
		for c_i,p_i in enumerate(policy):
			mdp_file.write('module car{}policy\n\n'.format(c_i))
			mdp_file.write('\tc{}_up{}: bool init false;\n'.format(c_i,c_i))
			mdp_file.write('\tc{}_down{}: bool init false;\n'.format(c_i,c_i))
			mdp_file.write('\tc{}_left{}: bool init false;\n'.format(c_i,c_i))
			mdp_file.write('\tc{}_right{}: bool init false;\n'.format(c_i,c_i))
			mdp_file.write("\t[up{}] c{}_up{} -> 1:(c{}_up{}'=false);\n".format(c_i,c_i,c_i,c_i,c_i))
			mdp_file.write("\t[down{}] c{}_down{} -> 1:(c{}_down{}'=false);\n".format(c_i,c_i,c_i,c_i,c_i))
			mdp_file.write("\t[left{}] c{}_left{} -> 1:(c{}_left{}'=false);\n".format(c_i,c_i,c_i,c_i,c_i))
			mdp_file.write("\t[right{}] c{}_right{} -> 1:(c{}_right{}'=false);\n".format(c_i,c_i,c_i,c_i,c_i))
			for state_i in p_i:
				pol_line = '\t[assign] !carpol{} '.format(c_i)
				for sub_num,sub_state in enumerate(state_i):
					if sub_num>=2*agents:
						if sub_num%2==0:
							pol_line += ' & (xObs={})'.format(sub_state)
						else:
							pol_line += ' & (yObs={})'.format(sub_state)
					elif sub_num%2 == 0:
						pol_line += ' & (xA{}={})'.format(int(sub_num/2),sub_state)
					else:
						pol_line += ' & (yA{}={})'.format(int(sub_num/2), sub_state)
				mdp_file.write(pol_line+'-> (c{}_'.format(c_i)+act_list[p_i[state_i][c_i]]+"{}'=true);\n".format(c_i))
			mdp_file.write('endmodule\n\n')

	mdp_file.write('label "Crash" = (Cr=true);\nlabel "Goal" = (T=true);\n\n')
	for i in range(agents):
		for j in range(size+1):
			mdp_file.write('label "xA{}_{}" = (xA{}={});\n'.format(i,j,i,j))
			mdp_file.write('label "yA{}_{}" = (yA{}={});\n'.format(i,j,i,j))
		mdp_file.write('label "up{}" = (ua{}=true);\n'.format(i,i))
		mdp_file.write('label "down{}" = (da{}=true);\n'.format(i, i))
		mdp_file.write('label "left{}" = (la{}=true);\n'.format(i, i))
		mdp_file.write('label "right{}" = (ra{}=true);\n'.format(i, i))
	for k in range(size+1):
		mdp_file.write('label "xO{}" = (xObs={});\n'.format(k,k))
		mdp_file.write('label "yO{}" = (yObs={});\n'.format(k,k))
	mdp_file.close()
	return fname

def stormpy_process_next_state(model,result,state):
	return int(re.findall('\\d+', str(model.states[state].actions[result.scheduler.get_choice(state).get_deterministic_choice()].transitions))[0])

def load_box(fname):
	prism_program = stormpy.parse_prism_program(fname)
	# print("Building model from {}".format(fname))
	formula_str = 'Pmax=? [!"Crash" U "Goal"]'
	properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
	model = stormpy.build_model(prism_program)
	return model,properties


def loader(fname):
	prism_program = stormpy.parse_prism_program(fname)
	print("Building model from {}".format(fname))
	formula_str = 'Pmax=? [!"Crash" U "Goal"]'
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

def run_sample(numiter,numsample,thres,direction,parameters,model,properties,agents):
	#storing the approximate satisfaction probability for each iteration
	move_ids = [m_s.id for m_s in model.states if len(m_s.actions)==1]
	action_points = [m_s.id for m_s in model.states if len(m_s.actions)==4*agents]
	action_labels = [['up{}'.format(i),'down{}'.format(i),'left{}'.format(i),'right{}'.format(i)] for i in range(agents)]

	counterarray= [0 for _ in range(numiter)]
	instantiator = stormpy.pars.PMdpInstantiator(model)
	start3 = time.time()
	data_out = dict()
	for iter in range(numiter):
		counter=0
		# for each sample
		for i in tqdm(range(int(numsample))):
			#sample parameters according to the region defined by
			# "Parameter synthesis for Markov models: Faster than ever" paper
			points = [dict() for i in range(agents)]
			rv = []
			policies = [dict() for i in range(agents)]
			for x in parameters:
				for i in range(agents):
					s = 0.5 #np.random.uniform(1e-5, 1-1e-5)
					points[i][x] = stormpy.RationalRF(s)
					rv.append(s)
			#check result
			results = []
			for point_i in points:
				rational_parameter_assignments = dict([[x, stormpy.RationalRF(val)] for x, val in point_i.items()])
				instantiated_model = instantiator.instantiate(rational_parameter_assignments)
				result = stormpy.model_checking(instantiated_model, properties[0],extract_scheduler=True)
				results.append(result)

			for s_i in action_points:
				tab_state = tuple()
				for i in range(agents):
					tab_x = [int(re.findall('(?<=\_).*',ss_i)[0]) for ss_i in model.states[s_i].labels if 'xA{}'.format(i) in ss_i]
					tab_y = [int(re.findall('(?<=\_).*',ss_i)[0]) for ss_i in model.states[s_i].labels if 'yA{}'.format(i) in ss_i]
					tab_state += (tab_x[0],) + (tab_y[0],)
				obs_x = [int(re.findall('(?<=O).*',ss_i)[0]) for ss_i in model.states[s_i].labels if 'xO'.format(i) in ss_i][0]
				obs_y = [int(re.findall('(?<=O).*',ss_i)[0]) for ss_i in model.states[s_i].labels if 'yO'.format(i) in ss_i][0]
				tab_state += (obs_x,) + (obs_y,)
				for i in range(agents):
					s = s_i
					while s not in move_ids:
						s = stormpy_process_next_state(model,results[i],s)
					pol_tup = tuple()
					for j in range(agents):
						act_id = model.states[s].labels.intersection(set(action_labels[j]))
						pol_tup += (action_labels[j].index(act_id.pop()),)
					policies[i].update({tab_state:pol_tup})

			# result = stormpy.model_checking(instantiated_model, properties[0]).at(instantiated_model.initial_states[0])
			#append the counter according to the spec
			gc_dtmc = build_model_grid(g_agents,g_size,g_obs_region,policies)

			prism_program = stormpy.parse_prism_program(gc_dtmc)
			dtmc_model = stormpy.build_parametric_model(prism_program)
			dtmc_params = dtmc_model.collect_probability_parameters()

			sT = 0.5 # np.random.uniform(1e-5, 1 - 1e-5)
			rational_parameter_assignments = dict([[dtmc_params.pop(), stormpy.RationalRF(sT)] ])
			combined_instantiator = stormpy.pars.PDtmcInstantiator(dtmc_model)
			instantiated_dtmc = combined_instantiator.instantiate(rational_parameter_assignments)
			result_dtmc = stormpy.model_checking(instantiated_dtmc, properties[0]).at(instantiated_dtmc.initial_states[0])
			data_out.update({tuple(rv):result_dtmc})

			if direction==True:
				if float(result.at(instantiated_model.initial_states[0]))>thres:
					counter=counter+1
			else:
				if float(result.at(instantiated_model.initial_states[0]))<thres:
					counter=counter+1

		counterarray[iter]=counter/((numsample)*1.0)
	t3 = time.time()
	print("Solver time :" + str(t3 - start3))
	return counterarray,data_out

def run_box(boxes,agents):
	comb = list(product(* [list(range(len(boxes)))] * agents))
	action_labels = [['up{}'.format(i), 'down{}'.format(i), 'left{}'.format(i), 'right{}'.format(i)] for i in
					 range(agents)]

	start3 = time.time()
	data_out = dict()
	for c_i in tqdm(comb):
		policies = [dict() for i in range(agents)]
		results = []
		for c_z in c_i:
			t_1 = time.time()
			fname = build_model_grid(agents=g_agents,size=g_size, obs_box=boxes[c_z],run_box=True)
			model,properties = load_box(fname)
			move_ids = [m_s.id for m_s in model.states if len(m_s.actions)==1]
			action_points = [m_s.id for m_s in model.states if len(m_s.actions)==4*agents]
			result = stormpy.model_checking(model, properties[0], extract_scheduler=True)


			for s_i in action_points:
				tab_state = tuple()
				for i in range(agents):
					tab_x = [int(re.findall('(?<=\_).*', ss_i)[0]) for ss_i in model.states[s_i].labels if
							 'xA{}'.format(i) in ss_i]
					tab_y = [int(re.findall('(?<=\_).*', ss_i)[0]) for ss_i in model.states[s_i].labels if
							 'yA{}'.format(i) in ss_i]
					tab_state += (tab_x[0],) + (tab_y[0],)
				obs_x = \
				[int(re.findall('(?<=O).*', ss_i)[0]) for ss_i in model.states[s_i].labels if 'xO'.format(i) in ss_i][0]
				obs_y = \
				[int(re.findall('(?<=O).*', ss_i)[0]) for ss_i in model.states[s_i].labels if 'yO'.format(i) in ss_i][0]
				tab_state += (obs_x,) + (obs_y,)
				for i in range(agents):
					s = s_i
					while s not in move_ids:
						s = stormpy_process_next_state(model, result, s)
					pol_tup = tuple()
					for j in range(agents):
						act_id = model.states[s].labels.intersection(set(action_labels[j]))
						pol_tup += (action_labels[j].index(act_id.pop()),)
					policies[i].update({tab_state: pol_tup})
			print('Synth Time {}'.format(time.time() - t_1))

		t_2 = time.time()

		gc_dtmc = build_model_grid(g_agents, g_size, g_obs_region, policies,True)
		prism_program = stormpy.parse_prism_program(gc_dtmc)
		dtmc_model = stormpy.build_model(prism_program)
		result_dtmc = stormpy.model_checking(dtmc_model, properties[0]).at(dtmc_model.initial_states[0])
		print(dtmc_model.nr_states)
		print('Verification {}'.format(time.time()-t_2))
		data_out.update({c_i: result_dtmc})


	t3 = time.time()
	print("Solver time :" + str(t3 - start3))
	return data_out

# fname = build_model_grid(g_agents,g_size,g_obs_region)
# param,model,props = loader(fname)
# numiter= 5
# numsample= 5
# threshold=0.85
# direction=True
# counter_array,data_out=run_sample(numiter,numsample,threshold,direction,param,model,props,g_agents)
data_out = run_box(g_boxes,g_agents)
# fig = plt.figure()
# ax = plt.axes(projection='3d')

carlines = [[] for a in range(g_agents)]
zline = []
for x in data_out:
	for x_i,y in enumerate(x):
		carlines[x_i].append(y)
	zline.append(data_out[x])

#Figure 1
carlines.append(zline)
np.savetxt('Grid_Out{}_{}.csv'.format(g_agents,g_size),np.array(carlines).T,delimiter=',')
#
# ax.scatter3D(carlines[0],carlines[1],zline,c=zline,cmap=plt.get_cmap('RdYlGn'))
# ax.set_xlabel(r'$Car~1 - Pedestrian~box$')
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['Correct','Shifted in x','Shifted in y'])
# ax.set_ylabel(r'$Car~2 - Pedestrian~box$')
# ax.set_yticks([0,1,2])
# ax.set_yticklabels(['Correct','Shifted in x','Shifted in y'])
# ax.set_zlabel(r'$Composed~System~with~true~box$')
# plt.show()
# plt.savefig('Grid{}_{}agents.png'.format(g_size,g_agents), bbox_inches='tight')
#
# if g_agents>2:
# 	plt.cla()
# 	ax.scatter3D(carlines[1],carlines[2],zline,c=zline,cmap=plt.get_cmap('RdYlGn'))
# 	ax.set_xlabel(r'$Car~2 - Pedestrian~box$')
# 	ax.set_xticks([0,1,2])
# 	ax.set_xticklabels(['Correct','Shifted in x','Shifted in y'])
# 	ax.set_ylabel(r'$Car~3 - Pedestrian~box$')
# 	ax.set_yticks([0,1,2])
# 	ax.set_yticklabels(['Correct','Shifted in x','Shifted in y'])
# 	ax.set_zlabel(r'$Composed~System~with~true~box$')
# 	plt.savefig('Grid{}_{}agentsB.png'.format(g_size, g_agents), bbox_inches='tight')
# 	plt.cla()
# 	ax.scatter3D(carlines[0],carlines[2],zline,c=zline,cmap=plt.get_cmap('RdYlGn'))
# 	ax.set_xlabel(r'$Car~1 - Pedestrian~box$')
# 	ax.set_xticks([0,1,2])
# 	ax.set_xticklabels(['Correct','Shifted in x','Shifted in y'])
# 	ax.set_ylabel(r'$Car~3 - Pedestrian~box$')
# 	ax.set_yticks([0,1,2])
# 	ax.set_yticklabels(['Correct','Shifted in x','Shifted in y'])
# 	ax.set_zlabel(r'$Composed~System~with~true~box$')
# 	plt.savefig('Grid{}_{}agentsC.png'.format(g_size, g_agents), bbox_inches='tight')


