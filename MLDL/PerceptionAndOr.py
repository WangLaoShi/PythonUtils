#实现向量之间的操作
class VectorOp(object):
    # 向量点乘
    def dot(x,y):
        return sum(map(lambda item:item[0] * item[1],zip(x,y)))
    # 向量加法
    def add(x,y):
        data = [d1 + d2 for (d1,d2) in zip(x,y) ]
        return data

    # 向量乘标量
    def mul(v,s):
        return [e * s
                for e in v
                ]
# 实现一个感知器
class Perception(object):
    # 初始化参数列表，偏置变量，以及激活函数，学习效率
    def __init__(self,input_num,activor,rate):
        self.activor = activor
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
        self.rate = rate
    # 输出参数，以及偏置变量
    def __str__(self):
        return "\n权重列表：{0}，偏置变量：{1}，学习速率：{2}".\
            format(self.weights,self.bias,self.rate)

    # 输入一个数据，计算感知器结果 ，使用激活函数转换为0或者1
    def predict(self,input_vect):
        temp =  VectorOp.dot(self.weights,input_vect) + self.bias
        return self.activor(temp)

    # 使用输入结合训练感知器
    def train(self,input_vecs,labels,iteration):
        # 迭代次数
        for i in range(iteration):
            # 使用全部数据训练感知器
            for (input_vec,label) in zip(input_vecs,labels):
                # 下面这个换成self.predict(input_vec) - label  结果居然不对
                delat =label - self.predict(input_vec)
                # 调整参数
                self.update_weight(input_vec,label,delat);

    # 调整参数
    def update_weight(self,input_vec,label,delta):
        #weights = weight + （x * rate * delta）
        self.weights = VectorOp.add(self.weights ,
                                    VectorOp.mul(input_vec,self.rate * delta)
                                    )
        self.bias += self.rate * delta
        ## 这是这段代码的灵魂
        print(self)

# 激活函数
def f(x):
    return 1 if x > 0 else 0
#----------------------------------分割线---------------------------------------------------------------------------------
#----------------------------------分割线---------------------------------------------------------------------------------
#----------------------------------分割线---------------------------------------------------------------------------------
p = Perception(2,f,0.1)
# and 运算的结果
# 期望的输出列表，注意要与输入一一对应
# [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
input_vecs = [(1,1),(1,0),(0,1),(0,0)]
labels = [1,0,0,0]
p.train(input_vecs,labels,10)  # 10次迭代

print(p.__str__())

print("1 and 1 = %s" % p.predict((1,1)))
print("1 and 0 = %s" % p.predict((1,0)))
print("0 and 1 = %s" % p.predict((0,1)))
print("0 and 0 = %s" % p.predict((0,0)))

print("------------下面来尝试一下or运算---------------")
input_vecs = [(1,1),(1,0),(0,1),(0,0)]
labels = [1,1,1,0]
or_p = Perception(2,f,0.1)
or_p.train(input_vecs,labels,10)
print(or_p.__str__())
print("1 or 1 = %s" % or_p.predict((1,1)))
print("1 or 0 = %s" % or_p.predict((1,0)))
print("0 or 0 = %s" % or_p.predict((0,0)))
print("0 or 1 = %s" % or_p.predict((0,1)))