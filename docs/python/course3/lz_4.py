
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

# test9()


def test10():
    M1 = np.random.rand(2,3)
    M2 = np.random.rand(3,4)
    res = np.empty((M1.shape[0],M2.shape[1]))

    for i in range(M1.shape[0]):
        for j in range(M2.shape[1]):
            item = 0
            for k in range(M1.shape[1]):
                item += M1[i][k] * M2[k][j]
                res[i][j] = item

    res2 = M1@M2

    print(M1)
    print(M2)
    print(res)
    print(res2)
    print(((res2 - res) < 1e-15).all() )

# test10()

def test11():
    a = np.arange(1,10).reshape(3,3)
    a1 = (1/a).sum(axis = 1)
    a2 = np.array([a1,a1,a1]).T
    b = a*a2
    print(b)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i][j] = a[i][j]*sum(1/a[i])
    
    print(b)
    

# test11()

def test12():
    np.random.seed(0)
    a = np.random.randint(10, 20, (8, 5))
    a_sum_row = a.sum(axis=1)
    a_sum_col = a.sum(axis=0)

    total_sum = np.sum(a)
    print(np.sum(a))
    print(a_sum_row )
    print(a_sum_col )

    b = np.empty((a.shape[0],a.shape[1]))

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b[i][j]=a_sum_row[i] * a_sum_col[j] /total_sum

    c = a - b
    d = np.sum(np.square(c) / c)

    print(d)

from time import *

# test12()

def test13():

    begin_time =time()
    np.random.seed(0)
    m, n, p = 100, 80, 50
    B = np.random.randint(0, 2, (m, p))
    U = np.random.randint(0, 2, (p, n))
    Z = np.random.randint(0, 2, (m, n))

    U = U.T
    a = np.zeros((1,n))

    for i in range(B.shape[0]):
        a = np.row_stack((a,((B[i] - U)**2).sum(axis=1)))
    
    a = a[1:,:]
    
    print(np.sum(a * Z))

    end_time = time()
    print(end_time - begin_time)
    
    
# test13()


def solution():
    begin_time =time()
    np.random.seed(0)
    m, n, p = 100, 80, 50
    B = np.random.randint(0, 2, (m, p))
    U = np.random.randint(0, 2, (p, n))
    Z = np.random.randint(0, 2, (m, n))

    L_res = []
    for i in range(m):
        for j in range(n):
            norm_value = ((B[i]-U[:,j])**2).sum()
            L_res.append(norm_value*Z[i][j])
    
    print(sum(L_res))
    end_time = time()
    print(end_time - begin_time)

# solution()



# 输入一个整数的 Numpy 数组，返回其中递增连续整数子数组的最大长度。
# 例如，输入 [1,2,5,6,7]，[5,6,7]为具有最大长度的递增连续整数子数组，
# 因此输出3；输入[3,2,1,2,3,4,6]，[1,2,3,4]为具有最大长度的递增连续整数子数组，
# 因此输出4。请充分利用 Numpy 的内置函数完成。（提示：考虑使用 nonzero, diff 函数）

def test14():
    a = [1,2,5,6,7]
    a = [3,2,1,2,3,4,6,5]
    b = np.diff(a)
    b[b!=1] =0
    s = ''.join(str(i) for i in b)
    s = s.strip('0')
    t = [int(i) for i in s.split('0')]
    print(t)
    print(len(str(np.max(t))) + 1)
    

test14()