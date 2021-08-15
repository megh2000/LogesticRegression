import numpy as np
y=np.array([[1,2],[3,4]])
k=[[5],[4]]
#p=np.where(y==1)
#print(k)
#x=[np.ones(3,1),y]
#print(x)
print(np.shape(y),np.shape(k))
x=y*k
print(x)