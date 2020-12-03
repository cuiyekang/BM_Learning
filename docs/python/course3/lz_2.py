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

leetcode_852([1, 3, 4, 5, 3 ])
leetcode_852([1, 3, 2 ])
