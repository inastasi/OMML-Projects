import Project_2_dataExtraction as pde
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(1861402)
xLabel2,yLabel2,xLabel4,yLabel4,xLabel6,yLabel6=pde.returnData()

yLabel2=np.array([1]*1000)
yLabel4=np.array([-1]*1000)
label2_data=np.append(xLabel2,yLabel2.reshape(1000,1),axis=1)
label4_data=np.append(xLabel4,yLabel4.reshape(1000,1),axis=1)


all_data=np.append(label2_data,label4_data,axis=0)
print(all_data.shape)
X = all_data[:,:-1]
Y = all_data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

print('*'*50)

'''
# EACH ROW IS A clothing item; if we reshape it to the matrix we get the pic

plt.imshow(X_train[0].reshape(28,28), interpolation='nearest')
plt.show()
'''
#for testing data we scale tthe test data using weights from training adta> or wtf aha
"""
Scaling data
"""
def scaleData(data,mean,std):
  scaledData=(data-mean)/std
  return scaledData

mean = np.mean(X_train)                        
std = np.std(X_train)
X_train = scaleData(X_train,mean,std)
X_test = scaleData(X_train,mean,std)


plt.imshow(X_train[0].reshape(28,28), interpolation='nearest')
plt.show()



"""
To train a SVM, you have to convert the labels of the two classes of interest into '+1' and '-1'.
+1 - class2
-1 - class4
"""