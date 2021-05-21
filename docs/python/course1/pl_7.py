def print_directory_contents(sPath):
   import os                                      
   for sChild in os.listdir(sPath):                
       sChildPath = os.path.join(sPath,sChild)
       if os.path.isdir(sChildPath):
           print_directory_contents(sChildPath)
       else:
           print(sChildPath)


# print_directory_contents("./docs")

def extendlist(val,list=[]):
    list.append(val)
    return list

list1=extendlist(10)
list2=extendlist(123,[])
list3=extendlist('a')

print("list1 = %s" % list1)
print("list2 = %s" % list2)
print("list3 = %s" % list3)

from collections import defaultdict

d = defaultdict()
d['fl']=123

print(d)