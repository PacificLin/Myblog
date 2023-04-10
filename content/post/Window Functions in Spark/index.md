---
author: Pacific
categories:
- Spark
- Python
date: "2021-04-02"
description: 
image: 
tags:
- Spark
- Python
- Pyspark
- Sparksql
title: Window Functions in PySpark
---

## functions in PySpark

一般我工作的 Spark 環境，可以使用 Hive | sparklyr | PySpark 來操作 Spark dataframe，這篇以 PySpark 的環境來示範操作 window functions 😮 

由於是 Spark 是使用分散運算，因此 Spark 的環境並無法使用 pandas 的 module 來處理資料，而一般 Spark SQL 支持的 function 在 PySpark 的環境有兩種

> 1. 內置函數或 UDF(user define function)：針對每一個 row 返回計算後的值
>
> 2. 聚合函數：例如 sum | count | max 等等，依據分組進行運算，合併組後返回計算的值

但以上的 function 無法滿足更為複雜的運算，例如無法同時針對一組進行操作，且針對每一個 row 返回其特定的值，一般我們稱做這種計算為 window function

在 `Spark 1.4` 之後，提供了 sql.windows 函數來解決以上的困擾

```python
from pyspark.sql import Window
```

以下來講解使用的方式和時機，假設我們有以下 table 叫 `temp.wm`

name：用戶名字

assets：資產部位

value：資產價值

| name  | assets | value  |
| ----- | ------ | ------ |
| Jason | stock  | 6500   |
| Jason | fund   | 10000  |
| Jason | bond   | 15000  |
| Mike  | fund   | 12000  |
| Mike  | bond   | 350000 |
| Mike  | stock  | 200000 |
| Julia | gold   | 100000 |
| Julia | fund   | 5000   |
| Julia | stock  | 100000 |

我們要找出每一個用戶所擁有的最高和第二高價值的資產項目，一般而言 SQL 寫法如下

```sql
SELECT a.*
FROM 
 (
  SELECT
   name,
   assets,
   value,
   dense_rank() OVER (PARTITION BY name ORDER BY value DESC) AS rank
  FROM temp.wm
 ) AS a
WHERE a.rank < 3	
```

如果對於分組後個別計算的 window functions 組成語法不熟悉，下面簡單說明

- partitionBy：分組，選擇分組的欄位
- orderBy：排序，按照該欄位並依照 function 排序
- dense_rank()：window function 的一種，依照 ORDER BY 所指定的欄位對於每個 row 返回值 1, 2, 3...

也可以參考以下連結的教學

[SQL Window Functions | Advanced SQL - Mode Analytics](https://mode.com/sql-tutorial/sql-window-functions/)

則產出的結果如下

| name  | assets | value  | rank |
| ----- | ------ | ------ | ---- |
| Jason | bond   | 15000  | 1    |
| Jason | fund   | 10000  | 2    |
| Mike  | bond   | 350000 | 1    |
| Mike  | stock  | 200000 | 2    |
| Julia | gold   | 100000 | 1    |
| Julia | stock  | 100000 | 1    |
| Julia | fund   | 5000   | 2    |

而在 Spark 裡如果不使用 window functions 處理，也無法使用 pandas 的情況下處理 data 就會相對複雜且不易讀，故以下則以 Pysaprk 透過 window function 來示範處理

```python
import sys
from pyspark.sql.window import Window
import pyspark.sql.functions as F
df = sqlContext.table("temp.wm")

window_spec = Window.partitionBy("name") \
	.orderBy(F.col("value").desc())
    
df2 = df.withColumn("rank", Window.rank().over(window_spec)) \
	.filter(df.rank < 3).show(6, truncate = False)
```

以下則是 PySpark 有支援的 window functions 列表，以及和 SQL 的對照

### window functions 列表

|                        | SQL                       | DataFrame API |
| ---------------------- | ------------------------- | ------------- |
| **Ranking functions**  | rank                      | rank          |
|                        | dense_rank                | denseRank     |
|                        | percent_rank              | percentRank   |
|                        | ntile                     | ntile         |
|                        | row_number                | rowNumber     |
| **Analytic functions** | cume_dist                 | cume_dist     |
|                        | first_value               | firstValue    |
|                        | last_value                | lastValue     |
|                        | lag(Column, offset: Int)  | lag           |
|                        | lead(column, offset: Int) | lead          |

以下針對最常搞混的的幾個 functions 做說明，並以 temp_wm 裡的用戶 Julia 做說明

* `row_number()`

從 1 開始，根據 ORDER BY 所指定的 column 做排序

| name  | assets | value  | row_number() |
| ----- | ------ | ------ | ------------ |
| Julia | gold   | 100000 | 1            |
| Julia | stock  | 100000 | 2            |
| Julia | fund   | 5000   | 3            |

* `rank()`

從 1 開始，根據 ORDER BY 所指定的 column 做排序

| name  | assets | value  | rank() |
| ----- | ------ | ------ | ------ |
| Julia | gold   | 100000 | 1      |
| Julia | stock  | 100000 | 1      |
| Julia | fund   | 5000   | 3      |

> rank 和 row_number 的差異在於如果 column 的值相同， row_number 並不會重複給值（１, 2, 3），但 rank 則會（1, 1, 3）

* `dense_rank()`

從 1 開始，根據 ORDER BY 所指定的 column 做排序

| name  | assets | value  | dense_rank() |
| ----- | ------ | ------ | ------------ |
| Julia | gold   | 100000 | 1            |
| Julia | stock  | 100000 | 1            |
| Julia | fund   | 5000   | 2            |

> rank 和 dense_rank 的差異在於如果 column 的值相同， rank 會跳過被重複的值所佔據的位子（１, 1, 3），但 dense_rank 則不會（1, 1, 2）

以下再示範稍微複雜的作法，如果我們想知道每個人的資產價值與其擁有最高價值的資產相差多少

```python
import sys
from pyspark.sql.window import Window
import pyspark.sql.functions as F
df = sqlContext.table("temp.wm")

window_spec = Window.partitionBy("name") \
	.orderBy(F.col("value").desc())

df2 = df.withColumn("max_value", Window.max("value").over(window_spec)) \
    .withColumn("value_diff", F.col("max_value") - F.col("value")) \ 
    .selct("name", "assets", "value", "value_diff").show(9, truncate = False)
```

利用 window function 計算出每個用戶的最大資產價值，再與其每個資產項目做相減 ​​

如果沒有使用 window functions，則需要透過 aggregation 函數計算出每個用戶的資產最大值，再透過 join 合併 dataframe 再計算之，因此 window functions 相對的簡潔且易讀

最後結果如下 🙄 

| name  | assets | value  | value_diff |
| ----- | ------ | ------ | ---------- |
| Jason | bond   | 15000  | 0          |
| Jason | fund   | 10000  | 5000       |
| Jason | stock  | 6500   | 8500       |
| Mike  | bond   | 350000 | 0          |
| Mike  | stock  | 200000 | 150000     |
| Mike  | fund   | 12000  | 338000     |
| Julia | gold   | 100000 | 0          |
| Julia | stock  | 100000 | 0          |
| Julia | fund   | 5000   | 95000      |

### ROW FRAME 時間窗口

- `partitionBy`：分組，所有的通過 rowsBetween 和 rangeBetween 切割出來的幀都是在分組的基礎上的

- `orderBy`：排序，這個比較好理解，就是按照那個字段排序

- `rowsBetween/rangeBetween` ：rowBetween 是當前行的前或者後幾行，rangeBetween 是針對 orderby 的值計算出來的範圍再和 orderby 比較來得到時間幀

  * rowsBetween 不關心確切的值。它只關心行的順序，並且在計算幀時採用固定數量的前後行
  * rangeBetween 計算框架時考慮值

  rowsBetween 語法為 rowsBetween(x, y)，其中 x, y 可以是數字，-n 表示向前數 n 行，n表示向後數 n 行

而 rowsBetween/rangeBetween 也可使用以下方式來表示

- `Window.unboundedPreceding` 表示當前行的無限行
- `Window.currentRow` 表示當前行
- `Window.unboundedFollowing` 表示當前行之後的無限行

在一次用 `temp` 的 table 做舉例，但數字稍微做調整如下

| name  | assets | value  |
| ----- | ------ | ------ |
| Jason | stock  | 6500   |
| Jason | fund   | 10000  |
| Jason | bond   | 15000  |
| Mike  | fund   | 12000  |
| Mike  | bond   | 350000 |
| Mike  | stock  | 100000 |
| Julia | gold   | 100000 |
| Julia | fund   | 5000   |
| Julia | stock  | 100000 |

`rowsBetween` 中的幀不依賴於 orderBy 子句。所以會依照分組後做獨立計算

```python
window_spec = Window.partitionBy('assets').orderBy('value').rowsBetween(Window.unboundedPreceding, Window.currentRow)
df.withColumn('RowsBetween', F.sum(df.value).over(window_spec)).show()
```

| name  | assets | value  | RowsBetween |
| ----- | ------ | ------ | ----------- |
| Jason | stock  | 6500   | 6500        |
| Julia | stock  | 100000 | `106500`    |
| Mike  | stock  | 100000 | `206500`    |
| Julia | fund   | 5000   | 5000        |
| Jason | fund   | 10000  | 15000       |
| Mike  | fund   | 12000  | 27000       |
| Jason | bond   | 15000  | 15000       |
| Mike  | bond   | 350000 | 365000      |
| Julia | gold   | 10000  | 10000       |

改用 `rangeBetween` ，可以發現產出的值會取決於 orderBy 子句，如果值相同，會計算所有相同值得所有行，因此相同的 value 在同一行會一次做計算

```
window_spec = Window.partitionBy('assets').orderBy('value').rangeBetween(Window.unboundedPreceding, Window.currentRow)
df.withColumn('RowsBetween', F.sum(df.value).over(window_spec)).show()
```

| name  | assets | value  | RowsBetween |
| ----- | ------ | ------ | ----------- |
| Jason | stock  | 6500   | 6500        |
| Julia | stock  | 100000 | `206500`    |
| Mike  | stock  | 100000 | `206500`    |
| Julia | fund   | 5000   | 5000        |
| Jason | fund   | 10000  | 15000       |
| Mike  | fund   | 12000  | 27000       |
| Jason | bond   | 15000  | 15000       |
| Mike  | bond   | 350000 | 365000      |
| Julia | gold   | 10000  | 10000       |



### 參考來源

[Introducing Window Functions in Spark SQL - The Databricks Blog](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html)

[SQL Window Functions | Advanced SQL - Mode Analytics](https://mode.com/sql-tutorial/sql-window-functions/)

[PySpark Window Functions — SparkByExamples](https://sparkbyexamples.com/pyspark/pyspark-window-functions/)

[SQL高級知識——OVER開窗函數的用法 - 每日頭條 (kknews.cc)](https://kknews.cc/zh-tw/code/66jnqzv.html)