# -*- coding:utf-8 -*
import re

def split_uri():
    myDat = []
    with open("../data/good-xss-10000.txt") as f:
        for line in f:
            #/discuz?q1=0&q3=0&q2=0%3Ciframe%20src=http://xxooxxoo.js%3E
            index=line.find("?")
            if index>0:
                line=line[index+1:len(line)]
                #print line
                tokens=re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29',line)
                #print "token:", tokens
                myDat.append(tokens)
        f.close()
    print "the length of myDat:", len(myDat)

import urllib
import urlparse
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
import HTMLParser


#处理参数的最小值
MIN_LEN = 6
#状态个数
N=10
SEN=['<','>',',',':','\'','/',';','"','{','}','(',')']

def ischeck(str):
    if re.match(r'^(http)',str):
        return False
    for i, c in enumerate(str):
        if ord(c) > 127 or ord(c) < 31:
            return False
        if c in SEN:
            return True
        #排除中文干扰 只处理127以内的字符
    return True

def etl(str):
    vers=[]
    for i, c in enumerate(str):
        c=c.lower()
        if   ord(c) >= ord('a') and  ord(c) <= ord('z'):
            vers.append([ord('A')])
        elif ord(c) >= ord('0') and  ord(c) <= ord('9'):
            vers.append([ord('N')])
        elif c in SEN:
           vers.append([ord('C')])
        else:
            vers.append([ord('T')])
    print vers
    return np.array(vers)


def main(filename):
    X = [[0]]
    X_lens = [1]
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            result = urlparse.urlparse(line)
            # url解码
            query = urllib.unquote(result.query)
            params = urlparse.parse_qsl(query, True)
            #print params
            for k, v in params:
                #if ischeck(v):
                vers = etl(v)
                X = np.concatenate([X, vers])
                X_lens.append(len(vers))

    #print X
    #print X_lens
    remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    remodel.fit(X, X_lens)
    return remodel

def test(remodel, filename):
    with open(filename) as f:
        for line in f:
            # 切割参数
            result = urlparse.urlparse(line)
            # url解码
            query = urllib.unquote(result.query)
            params = urlparse.parse_qsl(query, True)

            for k, v in params:
                #print ischeck(v)
                #if ischeck(v):
                vers = etl(v)
                print "vers:", vers
                pro = remodel.score(vers)
                # print  "CHK SCORE:(%d) QUREY_PARAM:(%s) XSS_URL:(%s) " % (pro, v, line)
                #if pro >= T:
                print  "SCORE:(%d) QUREY_PARAM:(%s) XSS_URL:(%s) " % (pro, v, line)


filename = "../data/xss-test.txt"
remodel = main(filename)
test(remodel, filename)