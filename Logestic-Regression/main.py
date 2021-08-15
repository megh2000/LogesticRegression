#This program predicts the probability of student getting admission into the university based on his/her #results

import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
import matplotlib.pyplot as plt

def read_data():
    df=pd.read_csv("/home/megha/Documents/Projects/ML-AndrewNg/ML-Python/Logestic-Regression/ex2data1.txt",sep=",",header=None)
    #print(df)
    x=np.array(df.iloc[:,0:2])
    y=np.array(df.iloc[:,2])
    return (x,y)

def plot_data(x,y):
    pos=x[np.where(y==1)]#copying all x values having y=1, now pos contains all the x values labeled y=1
    neg=x[np.where(y==0)]
    plt.scatter([neg[:,0]],[neg[:,1]],marker='+',c='red')
    plt.scatter(pos[:,0],pos[:,1],marker='^',c='black')
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(["not Admitted","Admitted"])

def sigmoid(X,theta):
    return 1.0/(1.0+np.exp(-1.0*(X @ theta)))

def find_cost(X,theta,y,m):
    J=0
    h=sigmoid(X,theta)
    #print("hypothesis:",h)
    epsilon = 1e-5 
    J=(1/m)*np.sum(-(y.T) @ np.log(h+epsilon) - (1-y).T @ np.log(1-h+epsilon))
    return J

def grad_descent(theta,m,X,y,n_iters):
    alpha=0.004
    cost=np.zeros((n_iters,1))
    y=np.reshape(y,(-1,1))
    for i in range(n_iters):                                                                
        h=sigmoid(X,theta)
        #print(np.shape(X[:,1].T))
        #print(np.shape(y))
        theta=theta-(alpha/m)*(X.T @ (h-y))
        J=find_cost(X,theta,y,m)
        cost[i]=J
        print("Cost:",J)
    # plt.figure(figsize=(20,16))
    # #plt.plot(range(n_iters),cost,'b.')
    # plt.scatter(range(n_iters),cost,linewidths=0.4)
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Cost function")
    # plt.show()
    
    return (theta,J)

def predict(x_test,theta):
    p=sigmoid(x_test,theta)
    if p>=0.50:
        p=1
    else:
        p=0
    return p
    

def main():
    x,y=read_data()
    plot_data(x,y)
    plt.show()
    #plt.close()
    [m,n]=np.shape(x)
    #plt.show()
    X=np.concatenate((np.ones((m,1)),x),axis=1)
    theta=np.zeros((n+1,1))
    J=find_cost(X,theta,y,m)
    print("Cost for initial theta([0,0,0]) value:",J)
    theta,J=grad_descent(theta,m,X,y,n_iters=1000000)
    print("theta and cost for line of best fit:")
    print("theta:",theta,"Cost:",J)
    plot_data(x,y)
    plot_x=np.array([min(X[:,1])-2,max(X[:,2])+2])
    plot_y=(-1/theta[2]) * (theta[1] * plot_x + theta[0])
    #print(plot_x,plot_y)
    plt.plot(plot_x, plot_y,'-g', label = "Decision_Boundary")
    plt.show()
    print("Let's predict the output for some test data")
    m1,m2=[float(i) for i in input("enter exam 1 score and exam 2 score").split()]
    p=predict([1,m1,m2],theta)
    if p==1:
        print("Yes, person will get admission into the university")
    else:
        print("No, person will not get admission into the university")


    






if __name__=="__main__":
    main()