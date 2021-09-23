
from nfa import NFA
from dfa import *
import numpy as np
from scipy import stats
import random
from tqdm import tqdm
import itertools

class MDP(NFA):
    def __init__(self, states=[], alphabet=set(), transitions=[],init=None,L=dict([])):
        # we call the underlying NFA constructor but drop the probabilities
        trans = [(s, a, t) for s, a, t, p in transitions]
        self.dtmc = True if len(alphabet)==1 else False
        super(MDP, self).__init__(states, alphabet, trans)
        # in addition to the NFA we need a probabilistic transition
        # function
        self._prob_cache = dict()
        for s, a, t, p in transitions:
            self._prob_cache[(s, a, t)] = p
        self._prepare_post_cache()
        self.init = init
        self.L = L
        
    def add_labels(self,L):
        self.L = L
        
    def add_init(self,init):
        self.init = init

    def prob_delta(self, s, a, t):
        return self._prob_cache[(s, a, t)]
    
    def get_prob(self,t):
        return self._prob_cache.get(t)

    def sample(self, state, action):
        """Sample the next state according to the current state, the action,  and
        the transition probability. """
        if action not in self.available(state):
            return None
        # N = len(self.post(state, action))
        prob = []
        for t in self.post(state, action):
            prob.append(self.prob_delta(state, action, t))

        next_state = list(self.post(state, action))[np.random.choice(range(len(self.post(state, action))),1,p=prob)[0]]
        # Note that only one element is chosen from the array, which is the
        # output by random.choice
        return next_state

    def set_prob_delta(self, s, a, t, p):
        self._prob_cache[(s, a, t)] = p

    def evaluate_policy_E(self,policy,R, epsilon = 0.001, gamma = 0.9):
        V1 = dict.fromkeys(self.states,0)
        while True:
            e = 0
            V = V1.copy()
            for s in self.states:
                if type(policy[s]) == set:
                    a= random.choice(list(policy[s]))
                else:
                    a=policy[s]
                V1[s]= sum([self.prob_delta(s,a,next_s)*(gamma*V[next_s] + R[s,a]) for next_s in self.post(s,a)])
                e = max(e, abs(V1[s] - V[s]))
            if e < epsilon:
                return V

    def expected_utility(self,a, s, U):
        "The expected utility of doing a in state s, according to the MDP and U."
        return sum([self.prob_delta(s,a,next_s) * U[next_s] for next_s in self.post(s,a)])

    def best_policy(self, U):
        """Given an MDP and a utility function U, determine the best policy,
        as a mapping from state to action."""
        pi = {}
        utility = {s:dict() for s in self.states}
        for s in self.states:
            for a in self.available(s):
                utility[s][a] = self.expected_utility(a,s,U)
            pi[s] = utility[s].keys()[utility[s].values().index(max(utility[s].values()))]
        return pi

    def T_step_value_iteration(self,R, T):
        """Solving an MDP by value iteration for T-step horizon"""
        U1 = dict([(s, 0) for s in self.states])
        self._prepare_post_cache()
        policy = dict([(s, set()) for s in self.states])
        t = T
        while t > 0:
            U = U1.copy()
            delta = 0
            for s in self.states:
                U1[s] = max([sum([self.prob_delta(s,a,s1) * (U[s1] + R[s, a,s1])
                                  for s1 in self.post(s, a)])]
                            for a in self.available(s))[0]
                delta = max(delta, abs(U1[s] - U[s]))
            t = t - 1
            # print(t)
        for s in self.states:
            Vmax = dict()
            for a in self.available(s):
                Vmax[a] = [sum([self.prob_delta(s,a,s1) * (U[s1] + R[s, a,s1])
                                for s1 in self.post(s, a)])][0]
            maxV = max(Vmax.values())
            for a in Vmax.keys():
                if maxV == Vmax[a]:
                    policy[s].add(a)
                # if maxV == Vmax[a] <= epsilon:
                #     policy[s].add(a)
        return U, policy

    # def E_step_value_iteration(self,R,
    #                     epsilon=0.1, gamma=0.9):
    #     U1 = dict([(s, 0) for s in self.states])
    #     while True:
    #         U = U1.copy()
    #         delta = 0
    #         for s in self.states:
    #             U1[s] = max([sum([self.prob_delta(s,a,next_s) * (gamma*U[next_s] + R[s,a,next_s]) for next_s in self.post(s,a)])
    #                                         for a in self.available(s)])
    #             delta = max(delta, abs(U1[s] - U[s]))
    #             print(delta)
    #         if delta < epsilon * (1 - gamma) / gamma:
    #              break
    #     policy = self.best_policy(U)
    #     return policy

    def write_to_file(self,filename,initial,targets=set()):
        file = open(filename, 'w')
        self._prepare_post_cache()
        file.write('|S| = {}\n'.format(len(self.states)))
        file.write('|A| = {}\n'.format(len(self.alphabet)))
        file.write('s0 = {}\n'.format(list(self.states).index(initial)))
        if len(targets)>0:
            stri = 'targets = ('
            for t in targets:
                stri += '{} '.format(t)
            stri = stri[:-1]
            stri+=')\n'
            file.write(stri)

        file.write('s a t p\n')
        for s in self.states:
            for a in self.available(s):
                for t in self.post(s,a):
                    file.write('{} {} {} {}\n'.format(list(self.states).index(s),a,list(self.states).index(t),self.prob_delta(s,a,t)))

    def construct_MC(self,policy,filename = None,randomness=0):
        transitions = []
        if filename != None:
            file = open(filename, 'w')
            file.write('|S| = {}\n'.format(len(self.states)))
            file.write('s t p\n')
        returntrans = dict([(s, t), 0.0] for s in self.states for t in self.states)
        self._prepare_post_cache()
        for s in self.states:
            transdict = dict([(s, t), 0.0] for t in self.post_all(s))
            for a in self.available(s):
                if a in policy[s]:
                    w = 1.0 / len(policy[s]) - randomness / (len(self.states) - len(policy[s]))
                else:
                    w = (1.0 / (len(self.states) - len(policy[s]))) * randomness
                # tempdict = dict([(s, a, t),0.0] for t in states)
                for t in self.post(s,a):
                    p = self.prob_delta(s,a,t)
                    transdict[(s, t)] += p * w
            for t in self.post_all(s):
                returntrans[(s,t)] = transdict[(s, t)]
                if filename != None:
                    file.write('{} {} {}\n'.format(s, t, transdict[(s, t)]))
        return returntrans

                    # transitions.append((s, t, transdict[(s, a, t)]))


    def E_step_value_iteration(self,R,sink,targstates,
                        epsilon=1e-3, gamma=0.8):
        policyT = dict([])
        Vstate1 = dict([])
        Vstate1.update({s: 0 for s in self.states})
        e = 1
        Q = dict([])
        while e > epsilon:
            Vstate = Vstate1.copy()
            for s in tqdm(self.states - sink- targstates):
                acts = self.available(s)
                optimal = -1000
                act = None
                for a in self.available(s):
                    Q[(s, a)] = sum([self.prob_delta(s, a, next_s) *
                                     (gamma*Vstate[next_s] + R[(s,a,next_s)])
                                     for next_s in self.post(s, a)])
                    if Q[(s, a)] > optimal:
                        optimal = Q[(s, a)]
                        act = a
                    else:
                        pass
                acts = set([])
                for act in self.available(s):
                    if Q[(s, act)] == optimal:
                        acts.add(act)
                Vstate1[s] = optimal
                policyT[s] = {random.choice(tuple(acts))}
            e = max(np.abs([Vstate1[s] -
                         Vstate[s] for s in self.states]))  # the abs error
            print(e)
        return Vstate1, policyT

    def max_reach_prob(self, target,sinks=set(),epsilon=0.1):
        """
        infinite time horizon
        Value iteration: Vstate[s] the maximal probability of hitting the
        target AEC within infinite steps.
        """
        policyT = dict([])
        Vstate1 = dict([])
        R = dict()
        Win = target
        NAEC = set(self.states) - Win

        Vstate1.update({s: 1 for s in list(Win)})
        Vstate1.update({s: 0 for s in list(NAEC)})
        policyT.update({s: self.available(s) for s in list(Win)})
        e = 1
        Q = dict([])
        while e > epsilon:
            Vstate = Vstate1.copy()
            for s in tqdm(set(self.states) - Win - sinks):
                acts = self.available(s)
                optimal = 0
                act = None
                for a in self.available(s):
                    Q[(s, a)] = sum([self.prob_delta(s, a, next_s) *
                                     Vstate[next_s]
                                     for next_s in self.post(s, a)])
                    if Q[(s, a)] >= optimal:
                        optimal = Q[(s, a)]
                        act = a
                    else:
                        pass
                acts = set([])
                for act in self.available(s):
                    if Q[(s, act)] == optimal:
                        acts.add(act)
                Vstate1[s] = optimal
                policyT[s] = acts
            e = abs(max([Vstate1[s] -
                         Vstate[s] for s in self.states]))  # the abs error
            print(e)
                # print "iteration: {} and the state
                # value is {}".format(t, Vstate1)
        for s in sinks:
            policyT[s] = {'stop'}
        return Vstate1, policyT

    def policyTofile(self,policy,outfile):
        file = open(outfile, 'w')
        file.write('policy = dict()\n')
        for s in self.states:
            x = -s[1]
            y = s[0]
            t = (s[2]-270)%360
            s2 = (x,y,t)
            if s not in policy.keys():
                file.write('policy[' + str(s2) + '] = stop\n')
            else:
                if 'stop' not in policy[s]:
                    file.write('policy[' + str(s2) + '] = \'' + policy[s].pop() + '\'\n')
                else:
                    file.write('policy['+str(s2)+'] = stop\n')
        file.close()

    def computeTrace(self,init,policy,T,targ = None):
        s = init
        trace = dict()
        t = 0
        trace[t] = s
        while t < T:
            # print('t = ', t, 'state = ', s)
            act = list(policy[s])[0]
            ns = self.sample(s,act)
            t += 1
            s = ns
            trace[t] = ns
            if ns == targ:
                return trace
        return trace

    def write_PRISM_explicit(self,fname):
        file = open(fname+'.tra', 'w')
        file.write('MDP\n')
        state_dict = {s:j for j,s in enumerate(self.states)}
        self.stormpy_statedict = {v: k for k, v in state_dict.items()}
        for s in self.states:
            if self.available(s):
                for a in self.available(s):
                    for t in self.post(s,a):
                        file.write('{} {} {} {}\n'.format(state_dict[s],a,state_dict[t],self.prob_delta(s,a,t)))
            else:
                file.write('{} {} {} {}\n'.format(state_dict[s],0, state_dict[s], 1))
        file.close()
        file_lab = open(fname+'.lab','w')
        file_lab.write('#DECLARATION\n')
        file_lab.write('init ')
        for l_i in set.union(*self.L.values()):
            file_lab.write('{} '.format(l_i))
        file_lab.write('\n')
        file_lab.write('#END\n')
        write_list = [[state_dict[self.init],['init']]]
        for s_i in self.L:
            if self.L.get(s_i):
                if 'init' not in self.L[s_i]:
                    write_entry = state_dict[s_i]
                    write_terms = []
                    for j in self.L[s_i]:
                        write_terms.append(j)
                    write_list.append([write_entry,write_terms])
        write_list.sort(key= lambda x:x[0])
        for w_i in write_list:
            file_lab.write('{}'.format(w_i[0]))
            for j_i in w_i[1]:
                file_lab.write(' {}'.format(j_i))
            file_lab.write('\n')
        file_lab.close()

    def productMDP(self,prod_MDP,label_pairs=dict()):
        if prod_MDP.dtmc:
            new_alphabet = self.alphabet
        else:
            new_alphabet = set(itertools.product(self.alphabet,prod_MDP.alphabet))

        new_states = list(itertools.product(self.states,prod_MDP.states))
        new_trans = []
        new_labels = labels = dict([])
        for s,a,t in self.transitions:
            p = self._prob_cache[(s,a,t)]
            for n_s,n_a,n_t in prod_MDP.transitions:
                n_p = prod_MDP._prob_cache[(n_s, n_a, n_t)]
                for e in new_alphabet:break
                if e is int():
                    new_trans.append(((s,n_s),a,(t,n_t),p*n_p))
                else:
                    new_trans.append(((s,n_s),(a,n_a),(t,n_t),p*n_p))
                if self.L.get(s):
                    new_labels.update({(s, n_s):set(self.L.get(s))})
                if prod_MDP.L.get(n_s):
                    if new_labels.get((s,n_s)):
                        new_labels[(s,n_s)].union(set(prod_MDP.L[n_s]))
                    else:
                        new_labels.update({(s,n_s):set(prod_MDP.L.get(n_s))})
        for l_p in label_pairs:
            if new_labels.get(l_p):
                new_labels[l_p].union(set(label_pairs[l_p]))
            else:
                new_labels.update({l_p:set(label_pairs[l_p])})


        return MDP(states=new_states,alphabet=new_alphabet,transitions=new_trans,init=(self.init,prod_MDP.init),L=new_labels)



