import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_train=pd.read_csv('linear_data_train.csv')
data_test=pd.read_csv('linear_data_test.csv')
X_train=np.matrix(data_train.iloc[:,0:2])
Y_train=np.matrix(data_train.iloc[:,2])
Y_train=Y_train.T

X_test=np.matrix(data_test.iloc[:,0:2])
Y_test=np.matrix(data_test.iloc[:,2])
Y_test=Y_test.T

def fit(X_train,Y_train):

    lr=0.01
    epochs=3
    N=X_train.shape[0]

    w=np.random.rand(2,1)
    b=np.random.rand(1,1)

    x1_range=np.arange(X_train[:,0].min(),X_train[:,0].max(),0.1)
    x2_range=np.arange(X_train[:,1].min(),X_train[:,1].max(),0.1)

    Error=[]
    for i in range(epochs):
        for n in range(N):
            y_pred=np.matmul(X_train[n],w)+b

            e=np.subtract(Y_train[n],y_pred)

            w+=lr*X_train[n].T*e
            b+=lr*e

            Y_pred=np.matmul(X_train,w)+b
            error=np.mean(np.abs(Y_train-Y_pred))
            Error.append(error)

            ax=plt.subplot(1,2,1,projection='3d')
            x1,x2=np.meshgrid(x1_range,x2_range)
            z=x1*w[0]+x2*w[1]+b

    c=Y_train*10+10
    ax.plot_surface(x1, x2, z, rstride=1,cstride=1, alpha=0.4)
    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c=c, marker='o')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    # plt.show()

    ax2=plt.subplot(1,2,2)
    ax2.set_title('Loss')
    x=np.arange(0,len(Error))
    ax2.plot(x,Error,marker='|')
    plt.pause(0.001)
    plt.show()
    return w,b


def predict(X_test):
    y_pred=np.matmul(X_test,w)+b
    return y_pred

def evaluate(X,Y):
    y_pred = np.matmul(X, w) + b
    y_predic = np.zeros(len(y_pred))
    for i ,test in enumerate(X):
        y_predic[i]=predict(test)
    y_predic[np.where(y_predic<0)] =-1
    y_predic[np.where(y_predic>0)] = 1
    accuracy = (y_predic == Y).sum() / len(Y)
    loss = np.mean(np.abs(np.subtract(Y, y_pred)))
    return loss, accuracy

w,b = fit(X_train,Y_train)

y_pred=predict(X_test)
loss, accuracy = evaluate(X_test, Y_test)
loss_train, accuracy_train = evaluate(X_train, Y_train)
print('error',loss, 'accuracy',accuracy)




