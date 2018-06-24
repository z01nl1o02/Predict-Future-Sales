import os,sys,pdb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from collections import defaultdict


class SHOP(object):
    def __init__(self, root = 'shops'):
        self.root = root
        return
    def calc(self, shop,item, month, delta=1):
        path = os.path.join(self.root, '%d.csv'%shop)
        df = pd.read_csv(path)
        df = df[ df.item_id == item ]
        df['flag'] = df.apply(lambda x :  x['month'] != month and x['month'] >= month - delta and x['month'] <= month + delta, axis=1)
        df = df[ df.flag == 1 ]
        return df['cnt'].mean()


class CLF(object):
    def fit(self, X,Y, defvals):
        data = defaultdict(dict)
        for x,y in zip(X,Y):
            shop,item,month = x
            if item not in data[shop]:
                data[shop][item] = []
            data[shop][item].append( (month,y))
        self.data = data
        self.defvals = defvals
        return self
    def default_value2(self,item,month,shop, delta=2):
        tmp = []
        for m in range(month - delta, month + delta + 1):
            if m == month:
                continue
            if m not in self.defvals[item]:
                continue
            for shop in self.defvals[item][m].keys():
                tmp.extend( self.defvals[item][m][shop] )
        if len(tmp) < 1:
            return 0.5
        if len(tmp) == 1:
            return tmp[0]
        if 0:
            xx = [k for k in range(len(tmp))]
            yy = tmp
            plt.plot(xx,yy,'bo',label='def 2')
            plt.legend()
            plt.show()
        tmp = reduce(lambda a,b:  a + b, tmp) / len(tmp)
        #tmp = sorted(tmp)[len(tmp)//2]
        return tmp
    def default_value(self, item, month, shop, delta = 2):
        tmp = []
        for m in range(month - delta, month + delta + 1):
            if m == month:
                continue
            if m not in self.defvals[item]:
                continue
            if shop not in self.defvals[item][m]:
                continue
            tmp.extend( self.defvals[item][m][shop] )
        if len(tmp) < 1:
            return 0.5 #self.default_value2(item, month, shop) # improvement
        if len(tmp) == 1:
            return tmp[0]
        if 0:
            xx = [k for k in range(len(tmp))]
            yy = tmp
            plt.plot(xx,yy,'rx',label='def 1')
            plt.legend()
            plt.show()
        tmp = reduce(lambda a,b:  a + b, tmp) / len(tmp)
        return tmp
    def predict(self,X,Y = None, delta = 2):
        res = []
        flags = []
        for kk,x in enumerate(X):
            shop,item,month = x
            if item == 4840 and 0:
                res.append(2)
                continue
            if item == 13381 and 0:
                res.append(20)
                continue
            if item not in self.data[shop]:
                res.append( self.default_value(item,month,shop) )
                flags.append( 1 )
                continue
            tmp = self.data[shop][item]
            tmp = filter(lambda d : d[0] != month and d[0] >= month - delta and d[0] <= month + delta, tmp)
            if len(tmp) < 1:
                res.append( self.default_value(item,month,shop) )
                flags.append(1)
                continue
            if len(tmp) < 2:
                res.append( tmp[0][1] )
                flags.append(2)
                continue
            resval = reduce(lambda a,b : (0,a[1] + b[1]), tmp)[1] / len(tmp)
            res.append(resval)
            flags.append(3)
        print '#flag1: ',len( filter( lambda x: x == 1, flags) ) * 1.0 / len(flags),
        print ' #flag2: ',len( filter( lambda x: x == 2, flags) ) * 1.0 / len(flags),
        print ' #flag3: ',len( filter( lambda x: x == 3, flags) ) * 1.0 / len(flags)

        return res

class TESTDATA(object):
    def __init__(self, path='data/test.csv'):
        df = pd.read_csv(path)
        self.df = df
        month = 34
        shops, items = df['shop_id'], df['item_id']
        X = []
        for shop, item in zip(shops, items):
            X.append( (shop, item, month) )
        self.X = X
        return
    @property
    def data(self):
        return self.X
    def save(self,predY,postfix=""):
        df = pd.DataFrame({'ID':self.df.ID, 'item_cnt_month':predY})
        df.to_csv('submission%s.csv'%postfix,index=False)

class DATA(object):
    def __init__(self, root = 'shops'):
        dfs = []
        for csv in os.listdir(root):
            shop = np.int64( os.path.splitext(csv)[0] )
            path = os.path.join(root,csv)
            df = pd.read_csv(path)
            df['shop'] = shop
            dfs.append(df)
        df = pd.concat(dfs)
        shops, items, months, cnts = df['shop'], df['item'],df['month'],df['cnt']
        self.X, self.Y = [], []
        defcnts = defaultdict(dict)
        for shop, item, month, cnt in zip(shops, items, months, cnts):
            if cnt > 20:
                cnt = 20
            if cnt < 0:
                cnt = 0
            self.X.append( (shop, item, month) )
            self.Y.append( cnt )
            if month not in defcnts[item]:
                defcnts[item][month] = {}
            if shop not in defcnts[item][month]:
                defcnts[item][month][shop] = []
            defcnts[item][month][shop].append( cnt )
        self.cnts = defcnts
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y)
        return 

    @property
    def defvals(self):
        return self.cnts

    @property
    def data(self):
        return self.X

    @property
    def label(self):
        return self.Y

def split_by_month(X,month):
    trainidx = []
    testidx = []
    for idx,x in enumerate(X):
        #shop,item,month = x
        if x[2] == month:
            testidx.append(idx)
        else:
            trainidx.append(idx)
    return trainidx, testidx


def predict():
    dataset = DATA()
    trainX,trainY,defval = dataset.data, dataset.label,dataset.defvals
    clf = CLF().fit(trainX,trainY,defval)
    testset = TESTDATA()
    testX = testset.data
    predY = clf.predict(testX)
    testset.save(predY)
    print 'done!'
    return

def train():
    dataset = DATA()
    X,Y,defval = dataset.data, dataset.label,dataset.defvals
    errors = []
    for month in range(33,30,-1):
        trainidx,testidx = split_by_month(X,month)
        trainX,trainY = X[trainidx],Y[trainidx]
        testX,testY = X[testidx],Y[testidx]
        clf = CLF().fit(trainX, trainY,defval)
        try:
            predY = clf.predict(testX,testY)
            #predY = [0.5 for k in testY]
        except Exception,e:
            print e
            continue
        mse = np.sqrt( mean_squared_error(testY, predY) )
        errors.append( mse )
        print mse
    errors = np.asarray(errors)
    print('error mean %f (%f)'%(errors.mean(), errors.std()))


if __name__=="__main__":
    train()
    predict()





