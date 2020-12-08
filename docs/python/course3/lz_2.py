# 列表lst 内容如下
# lst = [2, 5, 6, 7, 8, 9, 2, 9, 9]
# 请写程序完成下列操作：

# 1. 在列表的末尾增加元素15
# 2. 在列表的中间位置插入元素20
# 3. 将列表[2, 5, 6]合并到lst中
# 4. 移除列表中索引为3的元素
# 5. 翻转列表里的所有元素
# 6. 对列表里的元素进行排序，从小到大一次，从大到小一次

def test_one():
    lst = [2, 5, 6, 7, 8, 9, 2, 9, 9]
    lst.append(15)
    print(lst)
    lst.insert(int(len(lst)/2),20)
    print(lst)
    lst.extend([2, 5, 6])
    print(lst)
    lst.pop(3)
    print(lst)
    lst.reverse()
    print(lst)
    lst.sort()
    print(lst)
    lst.sort(reverse = True)
    print(lst)

#test_one()


# 问题描述：
# lst = [1, [4, 6], True]
# 请将列表里所有数字修改成原来的两倍

def test_two():
    lst = [1, [4, 6], True]
    print(lst)
    lst[0] = lst[0]*2
    lst[1][0] = lst[1][0]*2
    lst[1][1] = lst[1][1]*2
    print(lst)

# test_two()

# leetcode 852题 山脉数组的峰顶索引
# 如果一个数组k符合下面两个属性，则称之为山脉数组
# 数组的长度大于等于3
# 存在 i， i>0 且 i<len(k)-1， 使得 k[0]<k[1]<......<k[i-1]<k[i]>k[i+1]>....>k[len(k)-1]
# 这个 就是顶峰索引。
# 现在，给定一个山脉数组，求顶峰索引。
# 示例:
# 输入: [1, 3, 4, 5, 3]
# 输出: True，峰顶 5
# 输入：[1, 2, 4, 6, 4, 5]
# 输出：False

def check_repeat(lst):
    #查看列表是否有重复值
    for i in range(len(lst)-1):
        for j in range(i+1,len(lst)):
            if lst[i] ^ lst[j] == 0:
                return True
    return False

def leetcode_852(lst):
    #数量是否大于3个
    #循环列表成员（去掉第一和最后一个），查看成员前边的列表（包括当前成员）升序是否有变化（不能有重复值），查看成员后边的列表（包括当前成员）降序是否有变化（不能有重复值）
    if len(lst)<3:
        print("False")
        return
    for i in range(1,len(lst)-1):
        left = lst[0:i+1]
        right = lst[i:]
        if check_repeat(left)  or  check_repeat(right):
            continue
        left.sort()
        right.sort(reverse=True)
        #print(left , " | " , right)
        if left == lst[0:i+1] and right == lst[i:]:
            print("True",lst[i])
            return
    
    print("False")

# leetcode_852([1, 3, 4, 5, 3 ])
# leetcode_852([1, 3, 2 ])


# print((1,2)*2)
# print((1,)*2)
# print((1)*2)

# a,b,*_ = (1,2,3,4)
# print(a,b)

# 实现函数isdigit， 判断字符串里是否只包含数字0~9
def isdigit(string):
    if len(string) == 0:
        return False
    try:
        for a in string:
            temp = int(a)
    except ValueError:
        return False
    
    return True

def isdigit2(string):
    return string.isdigit()

# print(isdigit("123456789"))
# print(isdigit("12358dd"))
# print(isdigit(""))

# print(isdigit2("123456789"))
# print(isdigit2("12358dd"))
# print(isdigit2(""))

# 给定一个字符串 s ，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
# 示例:
# 输入: "babad"
# 输出: "bab"
# 输入: "cbbd"
# 输出: "bb"


def leetcode_5(lst):
    # 按照每个字符将字符串分成2段，对比2段相同的位数（倒序对比正序），将相同位数放到 列表 里，返回 列表中最长的一个
    # 字符串2段有两种情况，一种左字符串包含当前字符，一种左字符串不包含当前字符
    all = []
    if len(lst) < 2:
        return ""
    # if len(lst) == 2 and lst[0] == lst[1]:
    #     return lst
    for i in range(len(lst)):
        # if i==0 or i==len(lst)-1:
        #     continue
        left = lst[0:i]
        right = lst[i+1:]
        # print(left,right)
        all.extend(leetcode_5_sub(left,right,lst[i]))
        left = lst[0:i+1]
        right = lst[i+1:]
        # print(left,right)
        all.extend(leetcode_5_sub(left,right,''))
    
    if len(all) == 0:
        return all
    max_len = len(all[0])
    for item in all:
        if max_len<len(item):
            max_len = len(item)
    
    result = [] 
    for item in all:
        if len(item) == max_len:
            result.append(item)
    return result


def leetcode_5_sub(left,right,current):
    all = []
    min_len = len(left)
    if len(right)<len(left):
        min_len = len(right)
    if min_len == 0 :
        return all    
    for j in range(min_len):
        if left[-1 - j] != right[j] and j==0:
            break
        if left[-1 - j] != right[j]:
            all.append(left[len(left) - j:] + current + right[:j])
            break
        if j == min_len -1 and left[-1 - j] == right[j]:
            all.append(left[len(left) - j - 1:] + current + right[:j + 1])
    return all

# l1 = leetcode_5("bb")
# print(l1)
# l2 = leetcode_5("cbbd")
# print(l2)
# l3 = leetcode_5("babad")
# print(l3)
# l4 = leetcode_5("abcdefedcb")
# print(l4)
# l5 = leetcode_5("abacccccccccc")
# print(l5)


# dic = {
# 'python': 95,
# 'java': 99,
# 'c': 100
# }

# dic["java"] = 98
# print(dic)
# dic.pop("c")
# print(dic)
# dic["php"] = 90
# print(dic)
# print(list(dic.keys()))
# print(list(dic.values()))
# print("javascript" in dic)
# print(sum(list(dic.values())))
# print(max(list(dic.values())))
# print(min(list(dic.values())))
# dic1 = {"php":97}
# dic.update(dic1)
# print(dic)

# 有一个字典，保存的是学生各个编程语言的成绩，内容如下
# data = {
# 'python': {'上学期': '90', '下学期': '95'},
# 'c++': ['95', '96', '97'],
# 'java': [{'月考':'90', '期中考试': '94', '期末考试': '98'}]
# }
# 各门课程的考试成绩存储方式并不相同，有的用字典，有的用列表，但是分数都是字符串类型，请实现函
# 数 transfer_score(score_dict) ，将分数修改成int类型

def transfer_score():
    data = {
    'python': {'上学期': '90', '下学期': '95'},
    'c++': ['95', '96', '97'],
    'java': [{'月考':'90', '期中考试': '94', '期末考试': '98'}]
    }

    data["python"]["上学期"] = int(data["python"]["上学期"])
    data["python"]["下学期"] = int(data["python"]["下学期"])

    data["c++"][0] = int(data["c++"][0])
    data["c++"][1] = int(data["c++"][1])
    data["c++"][2] = int(data["c++"][2])

    data["java"][0]["月考"] = int(data["java"][0]["月考"])
    data["java"][0]["期中考试"] = int(data["java"][0]["期中考试"])
    data["java"][0]["期末考试"] = int(data["java"][0]["期末考试"])

    return data

# print(transfer_score()["python"])
# print(transfer_score()["c++"])
# print(transfer_score()["java"])

def transfer_score_common_dic(data):
    for key_sub in list(data.keys()):
        if isinstance(data[key_sub],str):
            data[key_sub] = int(data[key_sub])

def transfer_score_common():
    data = {
    'python': {'上学期': '90', '下学期': '95'},
    'c++': ['95', '96', '97'],
    'java': [{'月考':'90', '期中考试': '94', '期末考试': '98'}]
    }

    for key in list(data.keys()):
        if isinstance(data[key],dict):
            transfer_score_common_dic(data[key])
        if isinstance(data[key],list):
            for i in range(len(data[key])):
                if isinstance(data[key][i],str):
                    data[key][i] = int(data[key][i])
                if isinstance(data[key][i],dict):
                    transfer_score_common_dic(data[key][i])
    return data

# print(transfer_score_common()["python"])
# print(transfer_score_common()["c++"])
# print(transfer_score_common()["java"])

# a = (1,)
# b = {'x','y','z'}
# c = set(['A','B','A','B'])
# print(a,type(a))
# print(b,type(b))
# print(c,type(c))

# x = {6,7,8}
# y = {7,8,9}
# print(x ^ y)

# x = {'A','B','C'}
# y = {'B','C','D'}
# print(x & y)

# a = [1,3,2,4]
# print(max(a))
# print(min(a))
# a.sort()
# print(a)
# print(sorted(a,reverse=True))

# import math

# a = list(range(1,101))
# print(sum(a))
# a = [2,3,4,5]
# r = []
# for i in a:
#     r.append(math.pow(i,1/3))
# print(r)

# a = ['x','y','z']
# b = [1,2,3]

# z = zip(a,b)
# print(list(z))




