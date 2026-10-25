import torch

# torch.arange(start, end, step, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
x = torch.arange(0,1.2,0.2,dtype=torch.float32)  # 创建一个行向量 x
x.shape  # 获取 x 的形状
x.numel()  # 获取 x 的元素个数
y = x.reshape(2,3)  # 将 x 重塑为 2 行 3 列的矩阵

# print(x.shape,y.shape)
# print(x.numel(),y.numel())
# print(x,y)

z1 = torch.zeros(2,5)  # 创建一个 2 行 5 列的全零矩阵 z
z2 = torch.ones(2,5,dtype=torch.int32)  # 创建一个 2 行 5 列的全一矩阵 z
z3 = torch.full((2,3), 2)  # 创建一个 2 行 3 列的矩阵 z，所有元素都是 2
z4 = torch.rand(2,3)  # 创建一个 2 行 3 列的矩阵 z，所有元素都是 [0,1) 区间的随机数
z5 = torch.randn(2,3)  # 创建一个 2 行 3 列的矩阵 z，所有元素都是标准正态分布的随机数
# print(z1,z2,z3,z4)
torch.tensor([[1,2],[3,4]])
torch.exp(x)  # 对 x 中的每个元素求指数 e^x
#常见的标准算术运算符（+、-、*、/ 和 **）都可以被升级为按元素运算
#print(z4+y,z4-y,z4*y,z4/y,z4**y)  # 对 x 和 y 中的每个元素进行加、减、乘、除和幂运算

t1 = torch.cat([z1,z2],dim=0)  # 按行拼接 z1 和 z2
t2 = torch.cat((z1,z3),dim=1)  # 按列拼接 z1 和 z3
t3 = torch.stack([z1,z2],dim=0)  # 按行堆叠 z1 和 z2
# print("t1:",t1,"t2:",t2,"t3:",t3)

# print(torch.tensor([[1,2],[3,4]]) == torch.tensor([[1,2],[2,4]]))  # 对两个张量中的每个元素进行比较，返回一个布尔张量
t3.sum()  # 对 t3 中的所有元素求和会产生一个只有一个元素的张量

# 广播机制：
'''
当拿两个“形状（Shape）不一样”的张量做加减乘除时，系统会自动把小的那个张量“拉伸”变大，
直到它和大的张量形状一样，然后再进行计算.而且它只是在逻辑上拉伸，
并没有真正在内存里复制数据，极其省内存）。
广播的条件：
  1.补齐维度：如果两个张量的维度数量不一样，系统会自动在维度较少的那个张量前面补 1。
    比如：A 的形状是 (3,)，B 的形状是 (2, 3)。
    系统会在 A 前面补一个 1，把它看成的形状是 (1, 3)。
  2.比较条件：
  对齐之后，从最右边（最后一维）向左边逐个维度比较，必须满足以下两个条件之一，才能广播：
       2.1 两个维度的大小完全相等。
       2.2 其中一方的维度大小是 1。
       如果既不相等，也没有 1，直接报错 RuntimeError。
'''

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
# print(a + b)

# 索引和切片
# print(y[-1])  # 取 y 的最后一行
# print(y[0:2])  # 取 y 的前两行
y[0,1] = 100  # 将 y 的第 0 行第 1 列的元素改为 100
y[0:2,:] = 200  # 将 y 的前两行的所有元素改为 200
y[:,0] = 300    # 将 y 的所有行的第 0 列的元素改为 300
y[0:1,0:2] = 400  # 将 y 的第 0 行的前 2 列的元素改为 400

before = id(y)  
# y+=y
y[:] = y+y # 使用 += 和 [:] 不会改变 y 的内存地址
after = id(y)
# print(before == after)

# 转换为Numpy 张量和python标量
xx = x.numpy()
xxx = torch.tensor(xx)
# print(type(xx), type(xxx),xx,xxx)
# print(torch.tensor(3.5).item())  # 将只有一个元素的张量转换为python标量 



# pandas 数据预处理
import os
os.makedirs(os.path.join('.','data'),exist_ok=True)  # 在当前文件的目录下创建一个 data 文件夹
data_file = os.path.join('.','data','house_tiny.csv')
# with open(文件名，模式【w or r】)  as f: w: 写入文件，如果文件存在就覆盖，r: 读取文件，如果文件不存在就报错
with open(data_file,'w',encoding='utf-8') as f: 
    f.write('NumRooms,Alley,Price\n')  # 写入表头
    f.write('NA,Pave,127500\n')  # 写入数据
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd
data = pd.read_csv(data_file)  # 读取数据 在读取时，会自动把刚才文件里的文本 NA 识别并转换成了 NaN（Not a Number，真正的空值）。
print(data)

# 缺失值处理
inputs,outputs = data.iloc[0:,0:2],data.iloc[0:,2]
inputs = inputs.fillna(inputs.mean(numeric_only=True)) # 用平均值填充缺失值
inputs = pd.get_dummies(inputs,dummy_na=True)  
#pd.get_dummies 会自动寻找表格里的所有字符串列，然后把它们拆分成几列全是 0 和 1 的“哑变量”
# dummy_na=True 的意思是：“请把 NaN（空值）也当成一种独立的类别，给它单独建一列。”
# data = data.drop(columns=[data.isnull().sum().idxmax()]) # 删除包含最多 NaN 的列
print(inputs)

inputs,outputs = inputs.astype(float),outputs.astype(float) #先把 DataFrame 整体强制转换为 float 类型
X,Y = torch.tensor(inputs.values,dtype=torch.float32),torch.tensor(outputs.values,dtype=torch.float32)
print(X,Y)
print(X.T,len(X.T))

print(X.sum(),X.sum(axis=0),X.sum(axis=1),X.mean(),X.mean(axis=0),X.mean(axis=1)) 
# axis = 0 按列求和 axis = 1 按行求和
print(X.sum(axis=0,keepdims=True)) #keepdims=True 保留原来的维度
print(X.cumsum(axis=0),X*X) # cumsum 累加 当前位置加上前面所有位置的值
x=torch.tensor([1,2])
y= torch.tensor([3,2])
print(torch.dot(x,y)) # torch.dot 计算两个向量的点积
print(torch.sum(x*y)) # torch.sum 计算两个向量的点积

A = torch.arange(12).reshape(4,3)
x = torch.tensor([1,2,3]).T
Ax = torch.mv(A,X) # torch.mv 计算矩阵和向量的乘积
print(Ax)



