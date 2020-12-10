
def test1():

    a = [m + '_'+n for m in ['a','b'] for n in ['c','d']]
    print(a)

    L = [1, 2, 3, 4, 5, 6, 7]
    b = [i if i<=5 else 5 for i in L]
    print(b)

    c = [(lambda x : 2*x)(i) for i in range(5)]
    print(c)

    d = list(map(lambda x,y:str(x) + '_'+y,range(5),list("abcde")))
    print(d)

# test1()

def test2():

    L1, L2, L3 = list('abc'), list('def'), list('hij')

    print(list(zip(L1,L2,L3)))
    print(tuple(zip(L1,L2,L3)))

    for i,j,k in zip(L1,L2,L3):
        print(i,j,k)

    L = list('abcd')
    for index,value in enumerate(L):
        print(index,value)

    for index,value in zip(range(len(L)),L):
        print(index,value)

    print(dict(zip(L1,L2)))

    zipped = list(zip(L1,L2,L3))
    print(zipped)

    print(list(zip(*zipped)))

# test2()

import numpy as np

def test3():
    print(np.linspace(1,5,11))
    print(np.arange(1,5,2))

    print(np.zeros((2,3)))
    print(np.eye(3))
    print(np.eye(3,k=1))
    print(np.full((2,3),10))
    print(np.full((2,3),[1,2,3]))

# test3()

def test4():
    print(np.random.rand(3))
    print(np.random.rand(3,3))

    a,b = 5,15
    print((b-a)*np.random.rand(3) + a)

    print(np.random.randn(2))
    print(np.random.randn(2,2))

    sigma, mu = 2.5,3
    print(mu + np.random.randn(3) * sigma)

    print(np.random.randint(5,15,(2,3)))

    my_list = ['a', 'b', 'c', 'd']
    print(np.random.choice(my_list,2,replace=False,p =[0.1,0.7,0.1,0.1]))
    print(np.random.choice(my_list,(2,3)))

    print(np.random.permutation(my_list))

    np.random.seed(0)
    print(np.random.rand())
    np.random.seed(0)
    print(np.random.rand())


# test4()

def test5():
    print(np.zeros((2,3)).T)
    print(np.r_[np.zeros((2,3)),np.zeros((2,3))])
    print(np.c_[np.zeros((2,3)),np.zeros((2,3))])
    print(np.r_[np.array([0,0]),np.zeros(2)])
    print(np.c_[np.array([0,0]),np.zeros((2,3))])


# test5()

def test6():
    target = np.arange(8).reshape(2,4)
    print(target)
    print(target.reshape((4,2),order = 'C'))
    print(target.reshape((4,2),order='F'))
    print(target.reshape((4,-1)))

    target = np.ones((3,1))
    print(target)
    print(target.reshape(-1))
    

# test6()


def test7():
    target = np.arange(9).reshape(3,3)
    print(target)
    print(target[:-1,[0,2]])
    print(target[np.ix_([True, False, True], [True, False, True])])
    print(target[np.ix_([1,2], [True, False, True])])

    a = target.reshape(-1)
    print(a)
    print(a[a%2==0])

# test7()

def test8():
    a = np.array([-1,1,-1,0])
    print(np.where(a>0,a,5))
    a = np.array([-2,-5,0,1,3,-1])
    print(np.nonzero(a))
    print(np.argmax(a))
    print(np.argmin(a))
    a = np.array([0,1])
    print(a.any())
    print(a.all())
    a = np.array([1,2,3])
    print(a.cumprod())
    print(a.cumsum())
    print(np.diff(a))

    target = np.arange(5)
    print(target)
    print(target.max())
    print(np.quantile(target,0.5))

    target = np.array([1, 2, np.nan])
    print(target)
    print(target.max())
    print(np.nanmax(target))
    print(np.nanquantile(target,0.5))

    a1 = np.array([1,3,5,9])
    a2 = np.array([1,5,3,-9])
    print(np.cov(a1,a2))
    print(np.corrcoef(a1,a2))

    a = np.arange(1,10).reshape(3,-1)
    print(a)
    print(a.sum(axis=0))
    print(a.sum(axis=1))

# test8()

def test9():
    res = 3 * np.ones((3,3)) + 1
    print(res)
    print(1/res)

    a = np.ones((3,2))
    print(a)
    print(a* np.array([[3,2]]))
    print(a * np.array([[2],[3],[4]]))
    print(a * np.array([[2]]))

    print(np.ones(3))
    print(np.ones(3) + np.ones((2,3)))
    print(np.ones(3) + np.ones((2,1)))
    print(np.ones(1) + np.ones((2,3)))

    a = np.array([1,2,3])
    b = np.array([1,3,5])

    print(a.dot(b))

    martix_target =  np.arange(4).reshape(-1,2)
    print(martix_target)
    print(np.linalg.norm(martix_target, 'fro'))
    print(np.linalg.norm(martix_target, np.inf))
    print(np.linalg.norm(martix_target, 2))

    vector_target =  np.arange(4)
    print(vector_target)
    print(np.linalg.norm(vector_target, np.inf))
    print(np.linalg.norm(vector_target, 2))
    print(np.linalg.norm(vector_target, 3))

    a = np.arange(4).reshape(-1,2)
    b = np.arange(-4,0).reshape(-1,2)
    print(a@b)

test9()