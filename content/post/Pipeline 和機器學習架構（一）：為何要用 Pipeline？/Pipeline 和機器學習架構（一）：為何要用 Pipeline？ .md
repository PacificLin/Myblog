---
author: Pacific
categories:
- Machine Learning
- Python
date: "2023-06-15"
description: 
image: 
math: true
tags:
- Machine Learning
- Python
- Pipeline
- sklearn
- transformer
- estimator
title: Pipeline 和機器學習架構（一）：為何要用 Pipeline？ 
---



## 到底為何要學 Pipeline 呢？

---

為什麼要使用 `pipeline`呢?

> 因為好用 😎

我認為最最主要的原因就是防止 `data leakage 資料洩漏` ，再來就是讓整個機器學習的流水線更好維護也更好整理。

一開始自己學習 ML 時會很難理解 pipeline 到底要幹嘛，程式碼一邊分析一般劈哩啪啦就打下去一堆了，plot，feature engineering 等等等混在一起，其實真的很難維護。

首先先來說何謂 `data leakage 資料洩漏` 



## data leakage

---

簡單來說就是把 testing data 的資料洩漏到 training data 啦。

這是標準定義，但我認為另一種更容易犯錯，也就是將 transformer 分別 fit training data 和 testing data。



那這兩種 scenario  分別以程式碼來做舉例，以乳癌資料做示範為何會將 testing data 的資料洩漏出來。

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 導入乳癌資料集
data = load_breast_cancer()

# 轉換為 DataFrame
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target
df.columns = df.columns.str.replace(' ', '_')

X = df.drop(['target'], axis = 1)
y = df.target
```

將資料集做 `MinMaxScaler()` 的數據伸縮

```python
scaler = MinMaxScaler()
scaler.fit_transform(X)[:3, :3]
```

看一下前三個數據和特徵的狀態，已經是轉換成 0-1 之間的資料

|      | 0        | 1        | 2        |
| :--- | :------- | :------- | :------- |
| 0    | 0.521037 | 0.022658 | 0.545989 |
| 1    | 0.643144 | 0.272574 | 0.615783 |
| 2    | 0.601496 | 0.390260 | 0.595743 |



假如這時候才去分 training data 和 testing data，那就造成資料洩漏了。

因為 `MinMaxScaler()`這個 transformer 是依據整個位分割的數據架構來重新對數據做轉換，因此  testing data 的資料就洩漏給 training data 了，那做完這個轉換後，就很容易 over fitting

> testing data 要想成就只能用一次，不能依據 testing data 的好壞來調整模型，那這樣就會過擬合，應該在 training data 中切分 validation data 來調整資料

但這時候就會遇到一個問題，如果先切好 train 和 test data 但做一些 column 的轉換時，不就要 train 和 test 各做一次。

對! 就是要重複做，但用 pipeline 就不用喔 😎



```python
train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 22)

train_data['worst_smoothness'] = train_data['worst_smoothness'].apply(lambda x: x**2 if x > 10 else x)

test_data['worst_smoothness'] = test_data['worst_smoothness'].apply(lambda x: x**2 if x > 10 else (x - 3 if x < 5 else x))
```



如上就會變成 training data 和 teating data 各做一次，但有時候懶，就會先做完這些轉換後再去切割 data，其實就會搞得很混亂，程式碼也會比較不好讀。

接著就容易犯第二個錯，分別 fit 擬合 training data 和 testing data 並分開 transform



```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 創建示範數據
data = np.array([100, 200, 300, 10, 25, 50]).reshape(-1, 1)  # 原始數據


# 分割數據
train = data[:3]  # 第一部分數據假設為訓練集
test = data[3:]  # 第二部分數據假設為測試集

# 創建MinMaxScaler對象
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

# 對兩部分數據進行fit
scaler1.fit(train)
scaler2.fit(test)

# 對新數據進行transform
transformed_by_train = scaler1.transform(test) # 以 training data 做 transform
transformed_by_test = scaler2.transform(test) # 以 testing data 做 transform

print("Transformed data fit by train:\n{}".format(transformed_by_train))
print("Transformed data fit by test:\n{}".format(transformed_by_test))
```



這時候就會覺得很怪，為何又不能分開 fit 然後 分別 transform，從下面結果可以看的出來：

`fit with training data` ：竟然出現了負值，MinMaxScaler()  不是會轉換在 0 和 1 之間嗎? 

`fit with test data` ：看起來在 0-1 之間好像比較正常



```python
Transformed data fit by train:
[[-0.45 ]
 [-0.375]
 [-0.25 ]]
Transformed data fit by test:
[[0.   ]
 [0.375]
 [1.   ]]
```



`fit with test data`  看起來是對的，但實際上應該用 `fit with training data` ，因為模型是用 training data 的資料結構去訓練出來的，因此就應該用 training data 的資料架構去轉換，因此如果 testing data 的最大值或最小值超出 training data ，就有可能出現負數。

故會需要`依據資料架構做 transform` 的正確的做法就是 🫠



1. **要分別切開 training data 和 testing data**
2. **fit training data and transform testing data**



感覺很容易搞混，沒錯多錯幾次就知道了，都還沒分 validation set ，所以為了避免不知道何時切割資料和不知道何時需要轉換資料，pipeline 就可以來解決這個痛點啦。



## Transformer 轉換器 & Estimator 估計器

---

這邊先稍微講一下，上面一直在那邊 fit,  transform，到底在說啥？

pipeline 流水線裡，最要能先了解的就是 `transformer 轉換器 & estimator 估計器`，兩者都有 `fit.()`，其功用是返回學習完數據的物件

1. transformer 轉換器

transformer  轉換器是一種實現`transform`方法的估算器。以某種方式轉換數據的管道的一個物件 （object）。假設你對時間做 timestamp 轉換成 datetime 也可以包成一個 transformer，transformer`以某種方式轉換輸入資料 fit.(X)`，並針對需要轉換的新資料做 `transform.(需要轉換的資料)`

2. estimator 估計器

在 sklearn 裡大致上可以認為顧忌器就是某種預測器。其透過保留參數後和儲存學習後的資料物件，用於預測於新的值，，並且應該提供`set_params`和`get_params`，這部分要參考 `Baseestimator`。程式上`estimator 通過使用輸入數據 fit(X)，預測新的數據 predict.(y)`
