from pyBN import *
import itertools
import copy
import numpy as np
import time

from collections import OrderedDict
ID = 0
Index = 0
dic_L = {}
dic_T = {}
dic_D = {}
final_D = {}
temp_D = {}


def bn2PSDD(path, path2):
    # path1 BN, path2 vtree

    # ======read BN========
    global I, L, BN
    BN = read_bn(path)
    # ======read Vtree=====
    I, L, last_line, reverse_order = read_vtree(path2) #reverse_order is the node order from child

    print("PSDD transformation start")
    print(I)  # internal diction
    print(L)
    # =======convert start======================================7===============
    start_node = last_line.split()[1]
    SCTime = time.time()
    decision(start_node)
    print("--- Convert %s second---" %(time.time()-SCTime))
    # print("L node: ", dic_L)
    # print("T node: ", dic_T)
    # print("dic_D: ", dic_D)  # I dont care
    print("Index: ", Index)
    print("===========start to collect D==========")

    print("reverse_order ", reverse_order)
    collect_D(temp_D,reverse_order)
    print("===========collect D is done!==========")
    # print("final_D: ", final_D)
    # print("ct total: ", ID)

    write_psdd(path)

    print("The convert done!!!")
    # ======end convert=========================================================

def read_vtree(path):
    _L = {}
    _I = {}
    vtree_reverse=[]
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not (line.startswith('c')):
                if 'L' in line:
                    # print("1",'L: ',line)
                    new_vertex, variable = line.split()[1:]
                    _L[int(new_vertex)] = int(variable)
                    lastline = line
                elif 'I' in line:
                    # print("1",'I: ',line)
                    new_vertex, left_child, right_child = line.split()[1:]
                    _I[int(new_vertex)] = [int(left_child), int(right_child)]

                    lastline = line
                    vtree_reverse.append(int(new_vertex))
                if line == '':
                    break
    # print(vtree_reverse)
    return _I, _L, lastline, vtree_reverse


def decision(parent, evi={}, theta=0):
    global ID
    # print("parent: ",parent)
    print("tempID", Index)
    # print(evi)###
    print("1","===========enter decision node================ ", parent)
    parent = int(parent)
    prob, all_prime = cal_prob(evi, parent) #prob could be list
    # print(prob)
    if max(prob)==1:
        prob = max(prob)
        # print("plese check========================== ",prob)
    if parent in I:
        # =====calculate the probability for theta
        # =====judge the prob=1 for case AB, judge prob!=1 forloop
        if prob == 1:
            literal = str(all_prime[0])
            flag_literal = literal in evi
            LenAPrime = len(all_prime)
            if flag_literal:
                # print("1","decision && parent in I && literal in evi && prob=1")
                # print(evi)
                terminal(parent, evi, prob)
            elif not(flag_literal) and LenAPrime == 1:
                # print("1","decision && parent in I && literal not~~~~in evi && prob=1")
                # print(parent,evi)
                # print("all prime: ",all_prime)
                table = cpt(BN, evi, all_prime)   # to have probability table
                # table = cal_prob(evi,parent)[0]
                p_bool=str(1) if table.index(1)==0 else str(-1)
                p_var = str(all_prime[0])
                evi[p_var] = p_bool
                # print("heeee")
                terminal(parent, evi, prob)
                del evi[p_var]
            else:#rember need to chekc  all the prime in evi not only first one
                # print("error occur")
                table = cpt(BN, evi, all_prime) # to have joint probability table
                temp_lst = list(map(list, itertools.product([1, -1], repeat=LenAPrime)))
                indexVar = table.index(1)
                # still need to add evidence for no matter what kind of situation, two~n nodes
                for Prime,PBool in zip(all_prime,temp_lst[indexVar]):
                    p_var = str(Prime)
                    evi[p_var] = str(PBool)
                terminal(parent, evi, prob)
                for Prime in all_prime:
                    del evi[str(Prime)]

        else:
            # print("1","decision && parent in I && prob=theta")
            # create combination for terminal node
            var_list = combination(all_prime)  # !!!delete parent
            # print("1","combination", var_list)
            del_list = []
            for ver, p in zip(var_list, prob):  # ???add theta and skip theta=0
                if p==0:
                    pass
                else:# transfer from list to evidence dic
                    for voc in ver:
                        p_var = str(abs(voc))
                        del_list.append(p_var)
                        p_bool = str(int(voc / (abs(voc))))
                        evi[p_var] = p_bool
                    theta = p
                    terminal(parent, evi, theta)
                    # print("1","+++decide theta", theta,"+++decide evi", evi)
                    # print("!!!should hava a decision node!!!!!it is real", ct_ID)
            # should clean the evidence,for next terminal....AB, A notB
            del_list = list(set(del_list))
            for key in del_list:
                del evi[key]

    else:  # parent in literal
        # print("1","decision && parent in literal && prob=theta")
        # print("parent: ",parent,"===",evi)
        flag = parent==max(L)
        if flag:
            p_bool=str(1)
            p_var = str(L[parent])
            evi[p_var] = p_bool
            terminal(parent, evi, prob)
            del evi[p_var]
        else:
            terminal(parent, evi, prob)


def terminal(node, evi={}, theta=0):
    global ID,Index, final_D, Index
    if (node in I):
        # print("1","*********condition2",theta)
        prime = I[node][0]
        sub = I[node][1]
        decision(prime, evi, theta)
        decision(sub, evi, theta)

    elif not (node in I) and (theta == 1):  # literal and theta=1
        # print("1","*********condition3", "literal node")
        literal = L[node]
        # if str(L[node]) in evi: #get sign from evidence
        p_bool = int(evi[str(L[node])])
        #     true_flag = list(h2.cpt).index(0)  # f.cpt=[true,false]===>index=[0,1]
        #     if true_flag == 0: true_flag = -1
        newindex = literal * p_bool
        collect_L(ID, node, newindex, evi)

    elif not (node in I) and not (theta == 1): #print true node
        # print("1","*********condition4 TTTTrue node")
        literal = L[node]
        collect_T(ID, node, literal, theta[0], evi)


def combination(literals):
    """

    :param literals:
    :return: all the combination of literals
    """
    temp_len = len(literals)  # get the length
    temp_lst = list(map(list, itertools.product([1, -1], repeat=temp_len)))  # get the combination list
    lst = list(itertools.chain.from_iterable(temp_lst))  # take out nested the list[[]]
    var_list = (2 ** temp_len) * literals  # 2^l true and false
    vertex = list(map(lambda x, y: x * y, lst, var_list))  # no bracket to apply the function
    # add the bracket back
    new_list_vertex = []
    for i in range(int(len(vertex) / temp_len)):
        index = temp_len * i
        new_list_vertex.append(vertex[index:index + temp_len])

    return new_list_vertex


def cal_prob(evi, node):
    """
    when at decision node, need to know all probability from each child
    :param evi:evidence
    :param node:
    :return: probability list and all_prime node
    """
    true_flag = True  # default=true when it's literal like C or notC
    # cal_prob({},['1','2'])    # [0.48, 0.32000000000000006, 0.12, 0.08000000000000002]
    all_prime = []
    for x in L.keys():
        if int(x) <= node:
            all_prime.append(L[x])  # here should have all the prime smaller than parent
    for prime in all_prime:
        if str(prime) in evi:
            true_flag = True
        else:
            true_flag = False
            break

    if not (true_flag):  # conditional on P(A,B| ),P(C|A,B),P(D|A,B,C),P(E|A,B,C,D)
        # print("1","**********conditional probability")
        for lit in evi.keys():
            lit = int(lit)
            all_prime.remove(lit)
        # print("1",new_prime)
        # print("1",primes)
        cpt_value = cpt(BN, evi, all_prime)
        # print("1",cpt_value)
        prob = cpt_value
    elif true_flag:
        prob = [1]
        # print("1","probability should be 1")
        # print("1",prob)

    return prob, all_prime


def cpt(bn, evidence, vars):
    """
     0626 f.cpt return the cpt table not P(B)
    Condition : strictly follow[[1, 2], [1, -2], [-1, 2], [-1, -2]]
    <class 'list'>: [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
    Arguments
    value from conditional probability table

    :param bn:a BayesNet object
    :param evidence: a dictionary, where
        key = rv, value = instantiation
    :param vars:
    :return: joint probability table P(A,B,C)
    """
    # if all(str(keys) in evidence for keys in vars): # if all the vars in evidence, then prob=1
    #     final_cpt = [1]
    # else:
    BN_new = copy.copy(bn)
    new_sample = copy.copy(evidence)
    l_vars = len(vars)
    ct = 0
    for rv in vars:
        rv = str(rv)
        f = Factor(BN_new, rv)
        # reduce the sample is inconsistent with evidence
        for p in BN_new.parents(rv):
            if p in new_sample:
                f.reduce_factor(p, new_sample[p])

        if l_vars == 1:
            final_cpt = f
        else:
            # choice_probs = f.cpt
            choice_probs = f
            if ct == 0:
                final_cpt = choice_probs
                ct += 1
            else:
                # final_cpt = [x * y for x, y in itertools.product(final_cpt, choice_probs)]
                # final_cpt = [x * y for x in final_cpt for y in choice_probs]
                # print(Index)
                choice_probs.multiply_factor(final_cpt)
                final_cpt=choice_probs
    final_cpt = list(final_cpt.cpt)
    return final_cpt


def collect_L(int_ID, int_vtree, int_lit, int_evi):  # '''dic_L={ID:{'vtree': int_vtree,'literal': int_lit}}'''
    global dic_L, ID
    global Index, temp_D
    int_evi2 = copy.copy(int_evi) #need to know why copy evidence?
    test = {'vtree': int_vtree, 'literal': int_lit}
    if test in dic_L.values():
        for Id_real, v in dic_L.items():
            if v == test:
                test2 = {"ID": Id_real, 'vtree': int_vtree, 'literal': int_lit}, {'evi': int_evi2}
                temp_D[Index] = test2
                Index += 1
                break
    elif not (test in dic_L.values()):
        dic_L[ID] = test
        dic_D[ID] = test, {"evi": int_evi2}
        # Id_real = int_ID
        test2 = {"ID": ID, 'vtree': int_vtree, 'literal': int_lit}, {'evi': int_evi2}
        temp_D[Index] = test2

        ID += 1 #actuall is ct_ID
        Index += 1


def collect_T(int_ID, int_vtree, int_var, int_p, int_evi):
    '''dic_T={ID:{'vtree': int_vtree, 'var': int_var, 'theta':int_p }}'''
    global dic_T, ID
    global temp_D, Index
    int_evi2 = copy.copy(int_evi)
    test = {'vtree': int_vtree, 'var': int_var, 'theta': int_p}
    # test_temp = [{'vtree': int_vtree, 'var': int_var, 'theta': int_p}, {"evi":int_evi2}]
    if test in dic_T.values():
        for Id_real, v in dic_T.items():
            if v == test:
                test2 = {"ID": Id_real, 'vtree': int_vtree, 'literal': int_var}, {'evi': int_evi2}
                temp_D[Index] = test2
                Index += 1
                break

    elif not (test in dic_T.values()):
        dic_T[ID] = test
        dic_D[ID] = test, {"evi": int_evi2}  # I dont care
        # Id_real = int_ID
        test2 = {"ID": ID, 'vtree': int_vtree, 'literal': int_var}, {'evi': int_evi2}
        temp_D[Index] = test2
        ID += 1
        Index += 1
    # if not(test_temp in temp_D.values()):temp_D [int_ID] =


def collect_D(reference_D, order_list):
    """

    :param reference_D: dictionary before compression
    :param order_list: the top down node order from vtree
    :return: DecisionNodeDic
    """
    #oder list =['1', '7', '5', '3'], [3, 1, 13, 11, 9, 7, 5]
    global ID, final_D, Index

    root = order_list[-1]
    temp_loop_D = copy.deepcopy(reference_D)
    ct_test=0
    for head_node in order_list:
        s_time = time.time()
        prime = I[head_node][0]
        sub =  I[head_node][1]
        prime_n = [k for k in temp_loop_D.keys() if
                    temp_loop_D[k][0]['vtree']== prime ]
        sub_n = [k for k in temp_loop_D.keys() if
                 temp_loop_D[k][0]['vtree']== sub  ]
        evi_D_tailor_p = {}
        evi_D_tailor_s = {}

        test = root
        ancestors = [root]
        while test in I:
            if head_node < test:
                test = I[test][0]
                ancestors.append(test)
            elif head_node > test:
                test = I[test][1]
                ancestors.append(test)
            else:
                ancestors.append(test)
                break
        while test in I: test=I[test][0]
        prime_child = test
        all_sub = [L[x] for x in L if x >= prime_child]
        max_node = max(ancestors)
        n_node = len(set(ancestors))

        # ========maintain the evidence==================
        if len(sub_n) == len(prime_n):
            for i ,j in zip(sub_n, prime_n):
                evi_D_tailor_s[i] = copy.deepcopy(temp_loop_D[i])
                evi_D_tailor_p[j] = copy.deepcopy(temp_loop_D[j])
                if head_node > max_node: # delete evidence from sub
                    del_evi_sub = [str(x) for x in all_sub if str(x) in evi_D_tailor_s[i][1]['evi'] ]
                    for key in del_evi_sub:
                        del evi_D_tailor_s[i][1]['evi'][key]
                    del_evi_prime = [str(x) for x in all_sub if str(x) in evi_D_tailor_p[j][1]['evi'] ]
                    for key2 in del_evi_prime:
                        del evi_D_tailor_p[j][1]['evi'][key2]
                elif head_node < max_node:
                    pass
                elif head_node==max_node and n_node==1:
                    #only when collect then empty evidence
                    evi_D_tailor_s[i][1]['evi']={}
                    evi_D_tailor_p[j][1]['evi']={}
                elif head_node==max_node and n_node != 1:
                    del_evi_sub = [str(x) for x in all_sub if str(x) in evi_D_tailor_s[i][1]['evi'] ]
                    for key in del_evi_sub:
                        del evi_D_tailor_s[i][1]['evi'][key]
                    del_evi_prime = [str(x) for x in all_sub if str(x) in evi_D_tailor_p[j][1]['evi'] ]
                    for key2 in del_evi_prime:
                        del evi_D_tailor_p[j][1]['evi'][key2]
                    # same as case 1
                    # print("maintain pls")
                else: #in the case head_node==max_node, butn_node=!1
                    print("error in maintain evidence")

        else:
            print("error")
# ============maintain from here============================================================
        # start to collect with same length of evidence
        while evi_D_tailor_p:

            del_list=[] #index=Index
            # get minmum key in dic
            kk = min(evi_D_tailor_p)
            evi_test = evi_D_tailor_p[kk][1]
            # because when we generate node, its generate by combination order
            _temp_prime_list = [x for x in evi_D_tailor_p if evi_test == evi_D_tailor_p[x][1] ]
            _temp_prime_list.sort() # try to see evidence from temp_loop_D
            _temp_sub_list = [x for x in evi_D_tailor_s if evi_test == evi_D_tailor_s[x][1]]
            _temp_sub_list.sort()

# ============maintain from here============================================================
            evi = evi_test['evi']
            theta = cal_prob(evi, head_node)[0]
            theta = list(filter(lambda a: a != 0 ,theta))  ## !!!check!!!
            int_p_s_theta = []
            int_n = len(theta)

            for i in range(int_n):
                if int_n <= len(_temp_prime_list):
                    p = evi_D_tailor_p[_temp_prime_list[i]][0]['ID']
                    s = evi_D_tailor_s[_temp_sub_list[i]][0]['ID']
                    int_p_s_theta.extend([p, s, np.log(theta[i])])
                else:
                    print("====check====")
                    print("theta ",theta)
                    print("how many prime: ",len(_temp_prime_list))
            # get the vtree
            vtree_child = [temp_loop_D[_temp_prime_list[0]][0]['vtree'],
                           temp_loop_D[_temp_sub_list[0]][0]['vtree']]  # see the ID_number corresponding vtree
            int_vtree = [key for key, value in I.items() if value == vtree_child][0]  # for inference the vtree node
            test_final = {'vtree': int_vtree, 'n_el': int_n, 'p,s,theta': int_p_s_theta}

            for i,j in zip(_temp_prime_list,_temp_sub_list):
                del evi_D_tailor_p[i]
                del evi_D_tailor_s[j]
                del_list.extend([i,j])
            for i in del_list:
                del temp_loop_D[i]

            #==========add to final_D================
            if test_final in final_D.values():
                for Id_real, v in final_D.items():
                    if v == test_final:
                        test2 = {"ID": Id_real, 'vtree': int_vtree, 'literal': "com"}, evi_test
                        temp_loop_D[Index] = test2
                        Index += 1
                        break
            elif not (test_final in final_D.values()):
                final_D[ID] = test_final
                test2 = {"ID": ID, 'vtree': int_vtree, 'literal': "com"}, evi_test
                temp_loop_D[Index] = test2
                ID += 1
                Index += 1
        print("which vtree is done ", int_vtree)
        print("---%s th loop %s second---" %(ct_test,time.time()-s_time))
        ct_test += 1


def write_psdd(path):
    name = path.split("/")[-1]
    name = name.replace(".bif", ".psdd")
    address = '/home/austin/IdeaProjects/testpyBN/pyBN/data/psdd/' + name
    f1 = open(address, 'w')

    headline=('c ids of psdd nodes start at 0 \nc psdd nodes appear bottom-up, children before parents\nc\nc file syntax:\n' 
    'c psdd count-of-sdd-nodes \nc L id-of-literal-sdd-node literal\nc T id-of-trueNode-sdd-node id-of-vtree trueNode variable log(litProb)\n'
    'c D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub log(elementProb)}*\nc\n'
          'psdd %d\n' % ID)
    f1.write(headline)
    for i in range(ID):
        if i in dic_L:

            text = "L %s %s %s\n" % (str(i), str(dic_L[i]['vtree']), str(dic_L[i]['literal']))
        elif i in dic_T:
            text = "T %s %s %s %s\n" % (str(i), str(dic_T[i]['vtree']), str(dic_T[i]['var']), str(np.log(dic_T[i]['theta'])))
        elif i in final_D:
            text = "D %s %s %s %s\n" % (str(i), str(final_D[i]['vtree']), str(final_D[i]['n_el']), " ".join(map(str,final_D[i]['p,s,theta'])))

        f1.write(text)


    f1.close()

start_time = time.time()
BN_path = 'data/BN/32node.bif'
vtree_path = 'data/vtree/32node.vtree'
bbnode2 = bn2PSDD(BN_path, vtree_path)
print("--- Total %s second---" %(time.time()-start_time))
