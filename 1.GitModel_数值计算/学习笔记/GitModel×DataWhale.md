# GitModel×DataWhale

## 高等数学&数值计算（Day1-Day3）


<font color="orange">**Outline**</font>

- 最优化问题（极值、梯度）
- 插值问题（近似解、过数据点）
- 积分问题（连续问题离散化、累加求和）

#### Sympy

> Python的科学计算库，用以解决**多项式求值、求极限、解方程、求积分、微分方程、级数展开、矩阵运算**等问题。

##### 基础语法及函数

```python
# 导入库
from sympy import *

# 定义变量
x = sympy.Symbol('x')
n = symbols('n')

# 函数表达式
## 线性
fx = x*3+9
## 多元表达式
fx = x*y+2*x+1		

# evalf函数传值
y1 = fx.evalf(subs = {x:6})
result = fx.evalf(subs = {x:3,y:4})

# 方程求解
result2 = sovle(fx,x)
----------------------------------------------------------------------------

# 函数求导
## 一阶导数
n = symbols("n")
y = x*3+9
func1 = diff(y,n)
## m阶导数 
func2 = diff(y, n, m)

# 计算驻点
stag = solve(diff(y,n),n)

# 偏导
fx_y = diff(fx,y)
----------------------------------------------------------------------------

# 求和
fval = summation(x,(i,1,5))

# 求极限
lim1 = limit(f1,x,0)
lim2 = limit(f2,x,sympy.oo)

# 求定积分
result = integrate(f,(x,0,1))
----------------------------------------------------------------------------

# 数学符号补充
## 虚数单位i
sympy.I
## 自然对数e
sympy.E
## 无穷大
sympy.oo
## 圆周率
sympy.pi
## 求n次方根
sympy.root(8,3)
## 求对数
sympy.log(1024,2)
## 求阶乘
sympy.factorial(4)
## 三角函数
sympy.sin(sympy.py)
sympy.cos(sympy.pi/2)
sympy.tan(sympy.pi/4)
----------------------------------------------------------------------------

# 公式展开与折叠
f = (1+2*x)*x**2
fex = sympy.expand(f)
ffa = sympy.factor(f)

# 分数的分离和合并
f = (x+2)/(x+1)
fa = sympy.apart(f)
ft = sympy.together(f)

# 表达式简化
simplify((x**3+x**2-x-1)/(x**2+2*x+1))	#普通的化简
trigsimp(sin(X)/cos(X))				   #三角化简
powsimp(x**a*x**b)					   #指数化简
```

#### Scipy

> scipy包含各种专用于科学计算的工具箱，其不同的子模块对用不同的应用，如：**插值、积分、优化、图像处理、统计、特殊函数**。

| scipy.cluster     | 矢量量化/Kmeans                |
| ----------------- | ------------------------------ |
| scipy.constants   | 物理和数学常数                 |
| scipy.fftpack     | 傅里叶变换                     |
| scipy.integrate   | 积分                           |
| scipy.interpolate | 插值                           |
| scipy.io          | 数据输入输出                   |
| scipy.linalg      | 线性代数                       |
| scipy.ndimage     | n维图像包                      |
| scipy.odr         | Orthogonal distance regression |
| scipy.optimize    | 优化                           |
| scipy.signal      | 信号处理                       |
| scipy.sparse      | 稀疏矩阵                       |
| scipy.spatial     | 空间数据结构和算法             |
| scipy.special     | 任何特殊的数学函数             |
| scipy.stats       | 统计数据                       |

> 它们都依赖numpy，但大多相互独立。

##### 基础语法及函数

```python
#导入库
import scipy.optiminze as opt			# 导入优化子模块
from scipy import fmin 
import numpy as py

# 求多元极值
def func0(cost, x, a):					# 定义多元函数
    return cost*x*(2 - exp(-(x - a)**2))
func = lambda x: (2000*x[0] + 3000*x[1] + 4500*x[2]) / (func0(750, x[0], 6000) + func0(1250, x[1], 5000) + func0(2000, x[2], 3000)) - 1		# 所求多元表达式 
bnds = ((1, 10000), (1, 10000), (1, 10000))
res = opt.minimize(fun=func, x0=np.array([2, 1, 1]), bounds=bnds)
--------------------------------------------------------------------------

# 数值计算
from scipy import integrate # 已知函数表达式积分
from scipy import pi

def f(h):
    '''
    定义函数表达式.
    '''
    return 88.2 * pi * (5 - h)
v, err = integrate.quad(f, 0, 5) # 被积函数与积分区间
---------------------------------------------------------------------------

```

#### numpy

> python的科学计算库之一，高效地处理大型数据集。

##### 基础语法及函数

```python
import numpy as np
```



## Task——人口增长问题

<div class="alert alert-warning" role="alert">
<h4>💼 任务来袭</h4>

GitModel 公司对面试的实习生给出了这样一个问题 : 搜集 $1950\sim 2020$ 年间美国人口数据$,$ 猜测其满足的函数关系$,$ 并综合数据预测美国 $2030$ 年的人口数.

公司希望实习生就以下的<strong>开放性</strong>问题给出自己的想法，公司要求实习生<strong>提交预测的思路$,$ 模型$,$ 算法以及结果</strong>.

面试官给出了如下提示 : 预测值与真实值存在误差$,$ 该问题如何转化为可用上述所学的知识解决的问题呢? 


:key: **Solution**

未完待续......
