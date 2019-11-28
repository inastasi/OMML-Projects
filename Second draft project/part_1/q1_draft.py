import Project_2_dataExtraction as pde
from matplotlib import pyplot as plt

xLabel2,yLabel2,xLabel4,yLabel4,xLabel6,yLabel6=pde.returnData()
print(xLabel2.shape)
print(xLabel2[0].shape)
print(xLabel4[0].reshape(28,28).shape)
print('*'*50)


plt.imshow(xLabel4[0].reshape(28,28), interpolation='nearest')
plt.show()


"""
To train a SVM, you have to convert the labels of the two classes of interest into '+1' and '-1'.
"""