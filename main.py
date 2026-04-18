import numpy as np
from scipy.spatial.distance import cdist
import scipy
from scipy import linalg
from scipy.stats import ortho_group
import sys
from utils import *
from scipy.linalg import orth
from scipy.spatial.distance import pdist,squareform
import time
import os
from scipy.linalg import cho_factor, cho_solve



def run(X, W,alpha=1e1, beta=1, h=1, n_anchor=20):


    SMALL = 0

    n_views = len(X)
    n_samples = X[-1].shape[0]
    NList = list()
    for i_view in range(n_views):
        NList.append(X[i_view].shape[0])
    Vlist = list()
    Clist = list()
    Alist = list()
    n_tie = 200



    U = np.random.randn(n_samples,h)
    Lambda = np.ones(n_views)/n_views
    for i_view in range(n_views):
        Vlist.append(np.random.randn(X[i_view].shape[1], h))
        Clist.append(np.random.randn(X[i_view].shape[1], n_anchor))
        Alist.append(np.random.randn(n_anchor, h))

    loss1=0
    loss2=0
    loss3=0
    for i_view in range(n_views):
        loss1+=Lambda[i_view]*np.linalg.norm( U[:NList[i_view],:]-np.matmul(X[i_view], Vlist[i_view]))**2
        loss2+=Lambda[i_view]*alpha * np.sum(np.linalg.norm(Vlist[i_view], axis = 1))
        loss3+=Lambda[i_view]*beta * np.trace( np.matmul( Vlist[i_view].transpose(), np.matmul( (np.eye(Vlist[i_view].shape[0]) - np.matmul(np.matmul(Clist[i_view], np.diag(1/(np.sum(Clist[i_view], axis = 0)+SMALL))), Clist[i_view].transpose())), Vlist[i_view])))
    lossC=loss1+loss2+loss3
    lossP = -100


    for ite in range(n_tie):
        loss1=0
        loss2=0
        loss3=0
        for i_view in range(n_views):
            loss1+=Lambda[i_view]*(np.linalg.norm( U[:NList[i_view],:]-np.matmul(X[i_view], Vlist[i_view]))**2)
            loss2+=Lambda[i_view]*alpha * np.sum(np.linalg.norm(Vlist[i_view], axis = 1))
            for i_xx in range(Vlist[i_view].shape[0]):
                for j_xx in range(n_anchor):
                    loss3+=Lambda[i_view]*beta * Clist[i_view][i_xx,j_xx]*Clist[i_view][i_xx,j_xx]*np.linalg.norm(Vlist[i_view][i_xx,:] - Alist[i_view][j_xx,:] )**2
   
        lossP = lossC*1
        lossC=loss1+loss2+loss3
        if abs(lossP-lossC) <1e-4 and ite>10:
            break

        # update U
        for i_view in range(n_views):
            if i_view==0:
                U1 =  1/3*np.matmul(X[i_view][:NList[0],:],Vlist[i_view])
            else:
                U1 +=  1/3*np.matmul(X[i_view][:NList[0],:],Vlist[i_view])

        U2,_,U3 = np.linalg.svd(U1)
        U2 = U2[:, 0:h]
        UT = np.matmul(U2, U3)
        for i_view in range(n_views):
            if i_view==1:
                U1 =  1/2*np.matmul(X[i_view],Vlist[i_view])
            elif i_view==2:
                U1 +=  1/2*np.matmul(X[i_view],Vlist[i_view])
        U2,_,U3 = np.linalg.svd(U1)
        U2 = U2[:, 0:h]
        U = np.matmul(U2, U3)
        U[:NList[0],:] = UT 


        # update Ak
        for i_view in range(n_views):
            for i_sample in range(n_anchor):
                Alist[i_view][i_sample,:] = np.matmul( Clist[i_view][:,i_sample]*Clist[i_view][:,i_sample], Vlist[i_view] )/ np.sum(Clist[i_view][:,i_sample]*Clist[i_view][:,i_sample])


        # update Vk
        for i_view in range(n_views):
            V1 = np.matmul(X[i_view].transpose(), U[:NList[i_view],:])
            V2 = np.matmul(X[i_view].transpose(),X[i_view]) + alpha * np.diag( 0.5/ (np.linalg.norm(Vlist[i_view], axis = 1) +1e-12) ) + beta*(np.diag(np.sum( Clist[i_view]*Clist[i_view], axis = 1)) - np.matmul(np.matmul(Clist[i_view]*Clist[i_view], np.diag(1/(np.sum(Clist[i_view]*Clist[i_view], axis = 0)+SMALL))), (Clist[i_view]*Clist[i_view]).transpose()))
            c, low = cho_factor(V2)
            Vlist[i_view]   = cho_solve((c, low), V1)


        # update Ck
        for i_view in range(n_views):
            Sd =cdist(Vlist[i_view], Alist[i_view])

            Sd = 1/(Sd*Sd )
            Clist[i_view] = Sd / np.tile(np.sum(Sd,axis = 1).reshape(-1,1),(1,n_anchor))



    trueY=list()
    predY = list()
    for i_view in range(n_views):
        curY = list()
        for ww in W[i_view]:
            if abs(ww)>=0.1:
                curY.append(1)
            else:
                curY.append(0)
        # print(np.where(np.array(curY)>0))
        trueY.append(np.array(curY))        
        predY.append(np.linalg.norm( Vlist[i_view], axis = 1))


    return evaluate_AUC(trueY,predY),Vlist





X,W  = load_sysdata('simCFHData3_200_200_300_10')

W = (W[0],W[1],W[3])
X = (X[0],X[1],X[3])

auc, Vlist = run(X,W )
print(auc)

