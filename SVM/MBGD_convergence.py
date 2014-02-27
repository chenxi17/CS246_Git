import numpy as np
import pylab as pl
from time import time

'''
Implementing SVM with mini-batch gradient descent
Convergence criteria: exponentially weighted moving average of percentage change of cost function
'''

def costFunc(w,b,C,x,y):
    '''cost function for SVM with Hinge loss
    '''
    f=np.dot(w,w)/2
    for i in range(len(y)):
        f += C*max(0,1-y[i]*(b+np.dot(w,x[i,:])))
    return f

def trainSVM_MBGD(xTrain,yTrain,C,eta,errStop,tMax,nBatch):
    '''train SVM model using mini-batch gradient descent
    C: soft margin parameter
    nBatch: size of mini-batch,1: SDG, nData: batch DG
    eta: iteration step size
    errStop: convergence threshold
    tMax: maximum time step
    '''
    #shuffle data
    #combine x and y for shuffling
    xyTrain=np.concatenate((xTrain,np.reshape(yTrain,(len(yTrain),1))),axis=1)
    np.random.shuffle(xyTrain)
    xTrain=xyTrain[:,0:-1]
    yTrain=xyTrain[:,-1]

    nFeat=xTrain.shape[1] #number of features
    nData=xTrain.shape[0] #number of training examples

    #initialization
    w=np.zeros(nFeat)
    wNew=w
    b=0
    t=0 #iteration step count

    #cost function
    cost=[costFunc(w,b,C,xTrain,yTrain)]
    errList=[]

    #moving averaged error
    dCost=[0]
    dCostErr=100 #initial error
    L=0
    while dCostErr>errStop and t<tMax: #convergence criteria
        for j in range(nFeat):
            #df/dw
            grad=w[j]
            for i in range(L*nBatch,(L+1)*nBatch):
                i=np.mod(i,nData)
                if 1-yTrain[i]*(b+np.dot(w,xTrain[i,:]))>0:
                    grad += -C*yTrain[i]*xTrain[i,j]
            wNew[j]=w[j]-eta*grad
        #df/db
        grad=0
        for i in range(L*nBatch,(L+1)*nBatch):
            i=np.mod(i,nData)
            if 1-yTrain[i]*(b+np.dot(w,xTrain[i,:]))>0:
                grad += -C*yTrain[i]
        bNew=b-eta*grad
        #update
        b=bNew
        w=wNew
        
        #update cost function
        cost.append(costFunc(w,b,C,xTrain,yTrain))    
        if len(cost)>1:
            err=abs((cost[-2]-cost[-1])*100/cost[-2])
        errList.append(err)
        dCost.append(0.5*dCost[-1]+0.5*err)
        dCostErr=dCost[-1]        
        L+=1
        t+=1

    return w,b,cost


start=time()
#read data
yTrain=[]
with open('target.txt') as f:
    for line in f:
        yTrain.append(float(line))

xTrain=[]
with open('features.txt') as f:
    for line in f:
        line=line.strip()
        line=line.split(',')        
        xTrain.append(map(float,line))

yTrain=np.array(yTrain)
xTrain=np.array(xTrain)
nFeat=xTrain.shape[1] #number of features
nData=xTrain.shape[0] #number of training examples

#model parameters
C=100 #soft margin parameter
eta=0.001/C #time step
errStop=0.01 #convergence criteria
err=100
nBatch=20 #batch size, 1: SDG, nData: batch DG    
tMax=5000 #maximal iteration

w,b,cost=trainSVM_MBGD(xTrain,yTrain,C,eta,errStop,tMax,nBatch)

end=time()
print(end-start)

#plot cost function over time
pl.plot(range(len(cost)),cost,'*')
pl.text(1,590000,'Convergence time='+str(end-start)+'s')
pl.title('Mini-batch gradient descent')
pl.xlabel('Time step')
pl.ylabel('Cost function')
pl.show()
