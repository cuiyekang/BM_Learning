#电脑产生一个零到100之间的随机数字，然后让用户来猜，如果用户猜的数字比这个数字大，提示太大，否则提示太小，
#当用户正好猜中电脑会提示，"恭喜你猜到了这个数是......"。在用户每次猜测之前程序会输出用户是第几次猜测，如果用
#户输入的根本不是一个数字，程序会告诉用户"输入无效"

import random

def guessInt():

    guess = random.randint(1,101)

    i = 1 
    while True:
        print("第%d猜，请输入一个整数：" %(i))
        try:
            temp = int(input())
            i += 1
        except ValueError:
            print("输入无效")
            continue
        if temp == guess:
            print("猜对了")
            break
        elif temp>guess:
            print("猜大了")
        else:
            print("猜小了")

# 获取类的所有属性和方法

# print(dir(int))


# 原码：就是其二进制表示（注意，有一位符号位）。
# 反码：正数的反码就是原码，负数的反码是符号位不变，其余位取反（对应正数按位取反）。
# 补码：正数的补码就是原码，负数的补码是反码+1。
# 符号位：最高位为符号位，0表示正数，1表示负数。在位运算中符号位也参与运算。
# 输出2进制 
# 负数是以补码形式存储的
# print(bin(3))
# print(bin(-3))


# 通过位移快速计算

# n << 1 -> 计算 n*2
# n >> 1 -> 计算 n/2，负奇数的运算不可用
# n << m -> 计算 n*(2^m)，即乘以 2 的 m 次方
# n >> m -> 计算 n/(2^m)，即除以 2 的 m 次方
# 1 << n -> 2^n

# print(5 << 1)
# print(9 >> 1)
# print(5 << 2)
# print(8 >> 2)
# print(1 << 4)

#快速交换两个整数

# a,b = 2,3
# a ^= b
# b ^= a
# a ^= b
# print(a,b)


# leetcode 习题 136. 只出现一次的数字
# 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
# 尝试使用位运算解决此题

"""
循环每个数，与其他数做异或操作，数相同异或操作为0，只要有0就比较一下个数，注意不要跟自己比较了
"""

def leetcode136():
    
    t_list = [2,4,5,8,6,4,2,8,6]

    for index in range(len(t_list)):
        is_have_two = False
        for comp in range(len(t_list)):
            if(index + comp == len(t_list) - 1):
                break
            #print(t_list[index], t_list[-comp - 1 ])
            if t_list[index] ^ t_list[-comp - 1 ] == 0:
                is_have_two = True
                break
        if is_have_two == False:
            print("单个数:%d" %(t_list[index]))
            break


# 编写一个Python程序来查找那些可以被7除以5的整数的数字，介于1500和2700之间。

def testOne():

    result = []

    for i in range(1500,2701):
        if i % 7 ==0 and i%5==0:
            result.append(i)

    print(result)


# 2. 龟兔赛跑游戏
# 题目描述
# 话说这个世界上有各种各样的兔子和乌龟，但是 研究发现，所有的兔子和乌龟都有一个共同的特点——喜欢赛跑。
# 于是世界上各个角落都不断在发生着乌龟和兔子的比赛，小华对此很感兴趣，于是决定研究不同兔 子和乌龟的赛
# 跑。他发现，兔子虽然跑比乌龟快，但它们有众所周知的毛病——骄傲且懒惰，于是在与乌龟的比赛中，一旦任一秒
# 结束后兔子发现自己领先t米或以 上，它们就会停下来休息s秒。对于不同的兔子，t，s的数值是不同的，但是所有的
# 乌龟却是一致——它们不到终点决不停止。
# 然而有些比赛相当漫长，全程观看会耗费大量时间，而小华发现只要在每场比赛开始后记录下兔子和乌龟的数据——
# 兔子的速度v1（表示每秒兔子能跑v1 米），乌龟的速度v2，以及兔子对应的t，s值，以及赛道的长度l——就能预测
# 出比赛的结果。但是小华很懒，不想通过手工计算推测出比赛的结果，于是他找 到了你——清华大学计算机系的高才
# 生——请求帮助，请你写一个程序，对于输入的一场比赛的数据v1，v2，t，s，l，预测该场比赛的结果。
# 输入
# 输入只有一行，包含用空格隔开的五个正整数v1，v2，t，s，l，其中(v1,v2< =100;t< =300;s< =10;l< =10000且为v1,v2
# 的公倍数)
# 输出
# 输出包含两行，第一行输出比赛结果——一个大写字母“T”或“R”或“D”，分别表示乌龟获胜，兔子获胜，或者两者同
# 时到达终点。
# 第二行输出一个正整数，表示获胜者（或者双方同时）到达终点所耗费的时间（秒数）。

def rabbit_vs_tortoise(v1,v2,t,s,l):
    # 计算v2跑完全程所需要的时间，循环时间，计算当前时间v1，v2的距离，如果满足v1休息条件，则v1距离不变，v2增加距离。
    v2_total_time = int(l / v2)
    v1_current = 0
    v2_current = 0
    v1_total_time = v2_total_time + 1
    v1_Stop_s = s
    for i in range(1,v2_total_time + 1):
        v2_current = v2_current + v2 
        if v1_Stop_s < s:
            v1_Stop_s += 1
            continue
        v1_current = v1_current + v1
        #print(v1_current,v2_current,i)
        if v1_current >= l:
            v1_total_time = i
            break
        if v1_current - v2_current >= t:
            v1_Stop_s = 0
                

    if v1_total_time > v2_total_time:
        print("T",v2_total_time)
    elif v1_total_time == v2_total_time:
        print("D",v2_total_time)
    else :
        print("R",v1_total_time)
        

# rabbit_vs_tortoise(10,5,5,2,20)
# rabbit_vs_tortoise(10,5,5,3,20)
# rabbit_vs_tortoise(10,5,10,2,20)








