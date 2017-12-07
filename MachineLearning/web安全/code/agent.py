# -*- coding:utf-8 -*-
import sys
import urllib
import urlparse
import re
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
import HTMLParser
import nltk
import csv
import matplotlib.pyplot as plt

SEN=['<','>',',',':','\'','/',';','"','{','}','(',')']
FILE_MODEL="hworm.m"
def load_agent(filename):
    agent_list = []
    with open(filename, "r") as f:
        while True:
            agent = f.readline()
            if agent:
                #print agent
                agent_list.append(agent.strip())
            else:
                break
    return agent_list
# return agent_lists

def agent_to_avr(agent_list):
    vers = []
    for i in range(0, len(agent_list)):
        c = agent_list[i]
        if ord(c) >= ord('a') and ord(c) <= ord('z'):
            vers.append([ord('A')])
        elif ord(c) >= ord('0') and ord(c) <= ord('9'):
            vers.append([ord('N')])
        elif c in SEN:
            vers.append([ord('C')])
        else:
            vers.append([ord('T')])
    # print vers
    return np.array(vers)

def train_hmm(agent_lists):
    X = [[0]]
    X_lens = [1]
    for agent in agent_lists:
        ver=agent_to_avr(agent)
        X=np.concatenate([X,ver])
        X_lens.append(len(ver))
    remodel = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
    print "fitting......."
    remodel.fit(X, X_lens)
    joblib.dump(remodel, FILE_MODEL)
    print "done"
    return remodel

def test_alexa(remodel, filename):
    x = []
    y = []
    alexa_list = load_agent(filename)
    for agent in alexa_list:
        agent_ver = agent_to_avr(agent)
        pro = remodel.score(agent_ver)
        print  "SCORE:(%d) AGENT:(%s) " % (pro, agent)
        x.append(len(agent))
        y.append(pro)
    return x, y



def test():
    filename = "../data/useragent-1000.txt"
    hworm_filename = "../data/bad-useragent.txt"
    agent_lists = load_agent(filename)
    # print agent_lists[0]
    # print agent_to_avr(agent_lists[0])
    #remodel = train_hmm(agent_lists)
    remodel = joblib.load(FILE_MODEL)
    x_1, y_1 = test_alexa(remodel, hworm_filename)
    x_2, y_2 = test_alexa(remodel, filename)

    fig, ax = plt.subplots()
    ax.set_xlabel('Agent Length')
    ax.set_ylabel('HMM Score')
    ax.scatter(x_1, y_1, color='b', label="hworm")
    ax.scatter(x_2, y_2, color='g', label="normal")
    ax.scatter(x_1, y_1, color='r', label="alexa")
    ax.legend(loc='right')
    plt.show()


test()


