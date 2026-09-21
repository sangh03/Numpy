import numpy as np 


arr = np.array([1,2,3,4])

print(arr)

a = [1,2,3.5]


np.array(a)
print(a)

arr = np.arange(1,11,2)

print(arr)
arr = np.zeros((4,8))
print(arr)

arr = np.ones((6,6))
print(arr)
arr = np.linspace(0,1,100)
print(arr)

np.random.rand(10)

np.random.randn(10)

np.random.randint(10,20,10)

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr)


print(arr.shape)
print(arr.size)
print(arr.dtype)

print(arr.min())
print(arr.max())
print(arr.sum())
np.sum(arr,axis=1)

print(arr.mean())

print(arr.std())

print(arr.argmax())

print(arr.argmin())

arr=np.arange(1,31)
print(arr)
arr=arr.reshape(6,5)

print(arr)
