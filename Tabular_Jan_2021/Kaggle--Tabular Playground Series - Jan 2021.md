### Kaggle--Tabular Playground Series - Jan 2021

#### 1、检查空缺值

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_df = pd.read_csv("data/train.csv", index_col=0)
test_df = pd.read_csv("data/test.csv", index_col=0)
# 检查空缺值
missing_values = train_df.isnull().sum()

print(missing_values)
```

![image-20231127185247477](C:\Users\Morty\AppData\Roaming\Typora\typora-user-images\image-20231127185247477.png)

#### 2、检查相关性

热力图分析相关性，相关系数大于0.8或小于-0.8可以考虑删除一列

![](C:\Users\Morty\Downloads\无标题.png)

综合考虑把cont12删除

```python
X.drop('cont12', axis=1)
test_df.drop('cont12', axis=1)
```

#### 3、模型建立

##### 方法一：岭回归

```python
from sklearn.linear_model import RidgeCV
# 设置alpha值的范围
alphas = np.logspace(-6, 6, 13)

# 创建带交叉验证的岭回归模型
ridge_cv = RidgeCV(alphas=alphas, cv=5)  # cv代表交叉验证的折数

# 训练模型
ridge_cv.fit(X, Y)

# 输出最优的alpha值
print("Best alpha:", ridge_cv.alpha_) #结果为10
```

```python
# 创建岭回归模型
ridge = Ridge(alpha=1.0)  # alpha是正则化强度

# 训练模型
ridge.fit(X_train, Y_train)

# 预测
Y_pred = ridge.predict(X_test)

# 评估模型
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse) # 结果为0.523
```

提交后发现在Kaggle平台得分不行，大于为后10%

##### 方法2：随机森林

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
train_df = pd.read_csv("data/train.csv", index_col=0)
test_df = pd.read_csv("data/test.csv")

# 划分特征和目标变量
X = train_df.drop('target', axis=1)
Y = train_df['target']
X.drop('cont12', axis=1)
test_df.drop('cont12', axis=1)
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 创建XGBoost随机森林模型
xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=100, random_state=42)

# 训练模型
xgb_model.fit(X_train, Y_train)

# 预测测试集数据
Y_pred = xgb_model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error on Test Data with GPU:", mse)

predicted_targets = xgb_model.predict(test_df.drop('id', axis=1))

# 创建包含预测结果的DataFrame
results_df = pd.DataFrame({
    'id': test_df['id'],
    'target': predicted_targets
})

# 保存预测结果到CSV文件
results_df.to_csv('data/res.csv', index=False)
```

**使用XGBoost随机森林模型训练，发现成绩为：60%左右，明显提升**

##### 改进思路

xgboost重要性排行：（发现不能删）

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
train_df = pd.read_csv("data/train.csv", index_col=0)
test_df = pd.read_csv("data/test.csv")
from xgboost import plot_importance
import matplotlib.pyplot as plt

# 划分特征和目标变量
X = train_df.drop('target', axis=1)
Y = train_df['target']
X.drop('cont12', axis=1)
test_df.drop('cont12', axis=1)
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 创建XGBoost回归模型
model = xgb.XGBRegressor()

# 拟合模型
model.fit(X_train, Y_train)

# 计算特征重要性
importance = model.feature_importances_

# 特征重要性排行
feature_names = X_train.columns
feature_importance_ranking = sorted(zip(importance, feature_names), reverse=True)

# 打印特征重要性排行
for importance, feature in feature_importance_ranking:
    print(f"{feature}: {importance}")

# 可视化特征重要性
plot_importance(model)
plt.show()
```



![image-20231127205753619](C:\Users\Morty\AppData\Roaming\Typora\typora-user-images\image-20231127205753619.png)

集成（stacking）:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor
train_df = pd.read_csv("data/train.csv", index_col=0)
test_df = pd.read_csv("data/test.csv")
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# 划分特征和目标变量
X = train_df.drop('target', axis=1)
Y = train_df['target']
X.drop('cont12', axis=1)
test_df.drop('cont12', axis=1)
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 定义基模型
estimators = [
    ('ridge', Ridge(alpha=10.0)),
    ('xgb', xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=100, random_state=42)),
    ('catboost', CatBoostRegressor(iterations=100, random_state=42))
]

# 创建堆叠模型
stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# 训练堆叠模型
stack_model.fit(X_train, Y_train)

# 预测
Y_pred = stack_model.predict(X_test)

# 评估模型
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error with Stacking:", mse)

predicted_targets = stack_model.predict(test_df.drop('id', axis=1))

# 创建包含预测结果的DataFrame
results_df = pd.DataFrame({
    'id': test_df['id'],
    'target': predicted_targets
})

# 保存预测结果到CSV文件
results_df.to_csv('data/res.csv', index=False)
```

##### 进一步改进

寻找最优参数：

```python
# Ridge
# 设置alpha值的范围
alphas = [10.0, 11.0, 12.0, 13.0]

# 创建RidgeCV模型
ridge_cv = RidgeCV(alphas=alphas, cv=5)  # 使用5折交叉验证

# 训练模型
ridge_cv.fit(X_train, Y_train)

# 输出最优的alpha值
print("Best alpha:", ridge_cv.alpha_)

# 结果为12.0
```

```python
# XGBRegressor
# 参数分布
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.5, 0.7, 0.9, 1],
    'colsample_bytree': [0.5, 0.7, 0.9, 1]
}

# 创建XGBRegressor模型
xgb_model = XGBRegressor()

# 初始化随机搜索
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=25, cv=3, n_jobs=-1, verbose=2, random_state=42)

# 执行随机搜索
random_search.fit(X, Y)

# 最佳参数
print("Best parameters:", random_search.best_params_)

#结果为：Best parameters: {'subsample': 0.9, 'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'gamma': 0.2, 'colsample_bytree': 0.7}

```

这里是参数调整的详细说明：

- `n_estimators=200`：树的数量设置为200。
- `max_depth=8`：每棵树的最大深度设置为8。
- `learning_rate=0.05`：学习率设置为0.05。
- `gamma=0.2`：用于剪枝的最小损失减少设置为0.2。
- `subsample=0.9`：训练每棵树时用于随机采样的样本比例设置为90%。
- `colsample_bytree=0.7`：每棵树的训练时随机采样的特征比例设置为70%。
- `tree_method='gpu_hist'`：保持不变，使用GPU加速。
- `random_state=42`：为了结果的可重复性，保持随机状态不变。



```python
# CatBoostRegressor寻参数
# 参数范围
param_dist = {
    'depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': [100, 200, 300],
    'border_count': [32, 64, 128],
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'Poisson']
}

# 创建CatBoostRegressor模型
catboost_model = CatBoostRegressor()

# 随机搜索
random_search = RandomizedSearchCV(estimator=catboost_model, param_distributions=param_dist, n_iter=25, cv=3, scoring='neg_mean_squared_error', random_state=42)

# 执行随机搜索
random_search.fit(X, Y)

# 最佳参数
print("Best parameters:", random_search.best_params_)

# 结果：Best parameters: {'learning_rate': 0.1, 'l2_leaf_reg': 3, 'iterations': 300, 'depth': 8, 'border_count': 64, 'bootstrap_type': 'Bayesian'}
```

这里是参数调整的详细说明：

- `learning_rate=0.1`：学习率设置为0.1。
- `l2_leaf_reg=3`：L2正则化系数设置为3。
- `iterations=300`：树的数量设置为300。
- `depth=8`：树的最大深度设置为8。
- `border_count=64`：用于分割特征的边界数量设置为64。这个参数在GPU训练中特别重要，因为它影响模型的计算方式。
- `bootstrap_type='Bayesian'`：设置引导样本类型为贝叶斯。这与子样本采样方式有关。
- `task_type='GPU'`：保持不变，指定在GPU上运行。
- `random_state=42`：为了结果的可重复性，保持随机状态不变。

```python
# LGBMRegressor 最优参数模型
# 参数分布
param_dist = {
    'num_leaves': [33, 41, 51, 61],
    'max_depth': [3, 4, 5,  -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# 创建LGBMRegressor模型
lgbm_model = LGBMRegressor()

# 初始化随机搜索
random_search = RandomizedSearchCV(estimator=lgbm_model, param_distributions=param_dist, n_iter=25, cv=3, n_jobs=-1, verbose=2, random_state=42)

# 执行随机搜索
random_search.fit(X, Y)

# 最佳参数
print("Best parameters:", random_search.best_params_)

#最优结果：Best parameters: {'subsample': 1.0, 'num_leaves': 41, 'n_estimators': 200, 'max_depth': -1, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
```

这里是参数调整的详细说明：

- `subsample=1.0`：指定训练每棵树时用于随机采样的样本比例为100%。
- `num_leaves=41`：设定树模型中最大叶子的数量为41。
- `n_estimators=200`：树的数量设置为200。
- `max_depth=-1`：这个值为-1表示没有限制树的深度。
- `learning_rate=0.1`：学习率设置为0.1。
- `colsample_bytree=0.6`：每棵树的训练时随机采样的特征比例设置为60%。
- `random_state=42`：为了结果的可重复性，保持随机状态不变。

**最后stacking这几个模型，得到最后的模型，结果为rank:40%**



完整代码：

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor

train_df = pd.read_csv("data/train.csv", index_col=0)
test_df = pd.read_csv("data/test.csv")
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

# from sklearn.svm import SVR

# 划分特征和目标变量
X = train_df.drop('target', axis=1)
Y = train_df['target']
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 定义基模型
estimators = [
    ('ridge', Ridge(alpha=12.0)),
    ('xgb', xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=200, max_depth=8, learning_rate=0.05, gamma=0.2,
                             subsample=0.9, colsample_bytree=0.7, random_state=42))
    ('catboost',
     CatBoostRegressor(task_type='GPU', learning_rate=0.1, l2_leaf_reg=3, iterations=300, depth=8, border_count=64,
                       bootstrap_type='Bayesian', random_state=42)),

    ('lgbm', LGBMRegressor(subsample=1.0, num_leaves=41, n_estimators=200, max_depth=-1, learning_rate=0.1,
                           colsample_bytree=0.6, random_state=42))

    # ('svm', SVR())

]

# 创建堆叠模型
stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# 训练堆叠模型
stack_model.fit(X_train, Y_train)

# 预测
Y_pred = stack_model.predict(X_test)

# 评估模型
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error with Stacking:", mse)
predicted_targets = stack_model.predict(test_df.drop('id', axis=1))

# 创建包含预测结果的DataFrame
results_df = pd.DataFrame({
    'id': test_df['id'],
    'target': predicted_targets
})

# 保存预测结果到CSV文件
results_df.to_csv('data/res.csv', index=False)

```

