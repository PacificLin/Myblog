---
author: Pacific
categories:
- Machine Learning
- Python
- SQL
- data scientist
date: "2023-06-06"
description: 
image: 
math: true
tags:
- 找工作
- Python
- 刷題
- data scientist
- data analyst
title: Data Scientist 或 Data Analyst 找工作需要刷題嗎?
---

## 刷題的必要性

 DS 或 DA 不完全是工程師的導向，個人認為需要更多對於統計知識的普遍性理解（例如 `Bias-Variance Trade-Off`）。產業的 domain knowledge 以及商業邏輯的判斷和分析都是很重要的能力，DS 可能需要多一點對模型和統計方法的理解和應用，DA 需要更多的說故事能力以及資料視覺化（建立 dashboard 有要會 ）和商業的敏稅度。

相對於有 engineer 字尾的例如，algorithm engineer，ML engineer 或是 data engineer 需要更多的程式 solid 基礎，畢竟會需要考量系統效能，後端的穩定性，DevOps，測試等等，這些 position 可能需要刷比較多類演算法的題目。

Briefly, DS 和 DA  需要較強的`解決商業問題的能力`，而 engineer 結尾的需要比較強的程式建構能力。

當然我沒做過 engineer 結尾的啦 😐 😑



## 所以到底要不要刷

我之前找工作的經驗，會考寫程式的大概 5-7 成左右，其中大概有 8 成都是考 SQL，現場解的和帶回家做的都有，不過考 python 和 R 現場解的比較少。不過我也遇過當場給 data，然後要我現場分析後找出他們要的商業問題，一小時給你寫這樣，真的是滿硬的。

結論我覺得可以刷一下啦，但以 SQL 為主，Python 和 R 為輔。

但同時也需要練習一些 Non-coding Question lol，因為也滿多會問題模型原理和統計基礎的。



## 刷起來 !

這次育嬰離職快一年，隨然中間都還是有玩 Kaggle 和做公益專案，但都是用 Python，完全沒碰 SQL，但面試又幾乎都考 SQL ，所以既然要開始找工作了，那就認真刷一下題。

網路上資源其實滿多的，但很多都要付費，不過可以考慮先把免費的刷完再來看要不要付錢

看過 Reddit 推薦後，我主要決定刷以下幾個：

[LeetCode](https://leetcode.com/list/e97a9e5m/)

[stratascratch](https://www.stratascratch.com/)

[datalemur](https://datalemur.com/)

當然也可以找 Kaggle 的比賽來玩，但面試考程式不會考到你從 0 到建一個完整模型啦，通常都是用問的，因為建模其實要認真調校要花很多時間探索和反覆測試，光 cleaning data， EDA 和 feature engineering 就會花很多時間了。因此我認為刷題去熟悉資料處理的程序能力還是投資報酬率最大的。

以下我會記錄一下自己刷的過程中遇到滿有趣的題目，不太會分難易度，主要以自己可能比較不熟或覺得有趣為主



## Coding Question

---

### Users By Average Session Time

公司：Meta

難度：Medium

問題核心：table merge｜聚合（aggregation）

問題：

> Calculate each user's average session time. A session is defined as the time difference between a page_load and page_exit. For simplicity, assume a user has only 1 session per day and if there are multiple of the same events on that day, consider only the latest page_load and earliest page_exit, with an obvious restriction that load time event should happen before exit time event . Output the user_id and their average session time.



`facebook_web_log`

| user_id | timestamp           | action      |
| ------- | ------------------- | ----------- |
| 0       | 2019-04-25 13:30:15 | page_load   |
| 0       | 2019-04-25 13:30:18 | page_load   |
| 0       | 2019-04-25 13:30:40 | scroll_down |
| 0       | 2019-04-25 13:30:45 | scroll_up   |
| 0       | 2019-04-25 13:31:10 | scroll_down |
| 0       | 2019-04-25 13:31:25 | scroll_down |
| 0       | 2019-04-25 13:31:40 | page_exit   |

解法：

data 如表格，簡單來說就是一般在數位軌跡中的 session。

> session 指用戶一般在網站上登錄後到登出，或是未動作後超過一段時間。這個區間一般來說是 session，不過有些會定義一個 session 的最長時間，例如 30 分鐘，雖然你這 30 分鐘都沒登出，但超過 30 分鐘後，他會記錄為下一個 session

而他題目規定一個使用者一天就是一個 session，而著個 session 的計算方式為最後一個 log-in，但最先 log-out 的區間，

當然 log-in 的時間點要小於 log-out。

然後計算每個用戶的平均每天 session 長度

```python
# Import libraries
import pandas as pd
import numpy as np

# Start writing code
facebook_web_log.head()
df = facebook_web_log.copy()

# 1.先將 action 屬於 load 和 exit 的行為 filter 出來
df['timestamp'] = pd.to_datetime(df['timestamp'])
p_loads = df.loc[df['action'] == 'page_load', ['user_id', 'timestamp']]
p_exit = df.loc[df['action'] == 'page_exit', ['user_id', 'timestamp']]

# 2. 將兩個 table 合併，邏輯是每個 action 都去對應不童的 exit
sessions = pd.merge(p_loads, p_exit, how = 'left', on = 'user_id', suffixes = ['_load', '_exit'])

# 3. load 要小於 exit，filter 合理條件的 session
sessions = sessions[sessions['timestamp_load'] < sessions['timestamp_exit']]
sessions['timestamp'] = pd.to_datetime(sessions['timestamp_load'])

#4. 找出當日最後一次登錄還有最快離開的時間點
sessions = sessions.groupby(['user_id', pd.Grouper(key = 'timestamp', freq = 'D')]).agg({'timestamp_load': 'max', 'timestamp_exit': 'min'}).reset_index()

# 5. 計算 duration，這裡的 duration 就是 session 的時間
sessions['duration'] = sessions['timestamp_exit'] - sessions['timestamp_load']
sessions['duration'] = sessions['duration'].dt.total_seconds()

# 6.分組計算每個人平均每天的 session 時間
answer = sessions.groupby('user_id')['duration'].agg(lambda x: np.mean(x)).reset_index()
```



---

### Premium vs Freemium

公司：Microsoft

難度：Hard

問題核心：table merge｜聚合（aggregation）

問題：

> Find the total number of downloads for paying and non-paying users by date. Include only records where non-paying customers have more downloads than paying customers. The output should be sorted by earliest date first and contain 3 columns date, non-paying downloads, paying downloads.



`ms_user_dimension`

| user_id | acc_id |
| ------- | ------ |
| 1       | 716    |
| 2       | 749    |
| 3       | 713    |
| 4       | 744    |
| 5       | 726    |

`ms_acc_dimension`

| acc_id | paying_customer |
| ------ | --------------- |
| 700    | no              |
| 701    | no              |
| 702    | no              |
| 703    | no              |
| 704    | no              |
| 705    | no              |

`ms_download_facts`

| date                | user_id | downloads |
| ------------------- | ------- | --------- |
| 2020-08-24 00:00:00 | 1       | 6         |
| 2020-08-22 00:00:00 | 2       | 6         |
| 2020-08-18 00:00:00 | 3       | 2         |
| 2020-08-24 00:00:00 | 4       | 4         |
| 2020-08-19 00:00:00 | 5       | 7         |
| 2020-08-21 00:00:00 | 6       | 3         |
| 2020-08-24 00:00:00 | 7       | 1         |

解法：

1. 先將三張 table 根據 key 值 merge 
2. 針對有付費和沒付費德的用戶行為產生新的一列並將其改為 int
3. 聚合後加總，並將沒付費的大於有付費的日期 filter 出來

```python
# Import your libraries
import pandas as pd

# Start writing code
ms_user_dimension.head()

df = ms_user_dimension.copy()
df2 = ms_download_facts.copy()
df3 = ms_acc_dimension.copy()

df1 = df.merge(df3, how = 'left')
df_fnl = df2.merge(df1, how = 'left')
df_fnl.sort_values('date', ascending = True)

df_fnl['no'] = df_fnl.apply(lambda row: row['downloads'] if row['paying_customer'] == 'no' else 0, axis=1)
df_fnl['yes'] = df_fnl.apply(lambda row: row['downloads'] if row['paying_customer'] == 'yes' else 0, axis=1)

result = df_fnl.groupby('date').agg({'no':'sum', 'yes':'sum'}).reset_index()
result[result['no'] > result['yes']]
```

---



### Host Popularity Rental Prices

公司：Airbnb

難度：Hard

問題：雜湊 primary key | 聚合（aggregation）|

> You’re given a table of rental property searches by users. The table consists of search results and outputs host information for searchers. Find the minimum, average, maximum rental prices for each host’s popularity rating. The host’s popularity rating is defined as below: 0 reviews: New 1 to 5 reviews: Rising 6 to 15 reviews: Trending Up 16 to 40 reviews: Popular more than 40 reviews: Hot
>
> Tip: The `id` column in the table refers to the search ID. You'll need to create your own host_id by concating price, room_type, host_since, zipcode, and number_of_reviews.
>
> Output host popularity rating and their minimum, average and maximum rental prices.



`airbnb_host_searches`

| id      | price  | property_type | room_type       | amenities                                                    | accommodates | bathrooms | bed_type | cancellation_policy | cleaning_fee | city | host_identity_verified | host_response_rate | host_since          | neighbourhood     | number_of_reviews | review_scores_rating | zipcode | bedrooms | beds |
| ------- | ------ | ------------- | --------------- | ------------------------------------------------------------ | ------------ | --------- | -------- | ------------------- | ------------ | ---- | ---------------------- | ------------------ | ------------------- | ----------------- | ----------------- | -------------------- | ------- | -------- | ---- |
| 8284881 | 621.46 | House         | Entire home/apt | {TV,"Cable TV",Internet,"Wireless Internet","Air conditioning",Pool,Kitchen,"Free parking on premises",Gym,"Hot tub","Indoor fireplace",Heating,"Family/kid friendly",Washer,Dryer,"Smoke detector","Carbon monoxide detector","First aid kit","Safety card","Fire extinguisher",Essentials,Shampoo,"24-hour check-in",Hangers,"Hair dryer",Iron,"Laptop friendly workspace"} | 8            | 3         | Real Bed | strict              | TRUE         | LA   | f                      | 100%               | 2016-11-01 00:00:00 | Pacific Palisades | 1                 |                      | 90272   | 4        | 6    |
| 8284882 | 621.46 | House         | Entire home/apt | {TV,"Cable TV",Internet,"Wireless Internet","Air conditioning",Pool,Kitchen,"Free parking on premises",Gym,"Hot tub","Indoor fireplace",Heating,"Family/kid friendly",Washer,Dryer,"Smoke detector","Carbon monoxide detector","First aid kit","Safety card","Fire extinguisher",Essentials,Shampoo,"24-hour check-in",Hangers,"Hair dryer",Iron,"Laptop friendly workspace"} | 8            | 3         | Real Bed | strict              | TRUE         | LA   | f                      | 100%               | 2016-11-01 00:00:00 | Pacific Palisades | 1                 |                      | 90272   | 4        | 6    |
| 9479348 | 598.9  | Apartment     | Entire home/apt | {"Wireless Internet","Air conditioning",Kitchen,Heating,"Smoke detector","Carbon monoxide detector",Essentials,Shampoo,Hangers,Iron,"translation missing: en.hosting_amenity_49","translation missing: en.hosting_amenity_50"} | 7            | 2         | Real Bed | strict              | FALSE        | NYC  | f                      | 100%               | 2017-07-03 00:00:00 | Hell's Kitchen    | 1                 | 60                   | 10036   | 3        |      |

解法：

這題主要是要重新定義 host 的 primary key，因為 table 給你的 key 值是每次搜尋的，所以 host 會有重複，所以要重既有的 table 中雜湊出或拼出新的 key 值。

定義完了後根據題目條件針對不同的 host 做分類。

```python
# Import your libraries
import pandas as pd
import numpy as np
# Start writing code
airbnb_host_searches.head()

df = airbnb_host_searches.copy()

df['host_id'] = df['price'].map(str) + df['room_type'].map(str) + df['host_since'].map(str) + df['zipcode'].map(str)+ df['number_of_reviews'].map(str)

df = df.drop_duplicates(subset = 'host_id')

def review_rating(num):
    if num == 0:
        return 'New'
    elif num <= 5:
        return 'Rising'
    elif num <= 15:
        return 'Trending Up'
    elif num <= 40:
        return 'Popular'
    else:
        return 'Hot'
        
df['host_popularity'] = df['number_of_reviews'].apply(review_rating)
df = df.groupby('host_popularity').agg(min_price = ('price', min), avg_price = ('price', np.mean), max_price = ('price',max)).reset_index()
```

---



### Marketing Campaign Success [Advanced]

公司：Amazon

難度：Hard

問題核心：條件邏輯篩選

問題：

> You have a table of in-app purchases by user. Users that make their first in-app purchase are placed in a marketing campaign where they see call-to-actions for more in-app purchases. Find the number of users that made additional in-app purchases due to the success of the marketing campaign.
>
> The marketing campaign doesn't start until one day after the initial in-app purchase so users that only made one or multiple purchases on the first day do not count, nor do we count users that over time purchase only the products they purchased on the first day.



`marketing_campaign`

| user_id | created_at          | product_id | quantity | price |
| ------- | ------------------- | ---------- | -------- | ----- |
| 10      | 2019-01-01 00:00:00 | 101        | 3        | 55    |
| 10      | 2019-01-02 00:00:00 | 119        | 5        | 29    |
| 10      | 2019-03-31 00:00:00 | 111        | 2        | 149   |
| 11      | 2019-01-02 00:00:00 | 105        | 3        | 234   |
| 11      | 2019-03-31 00:00:00 | 120        | 3        | 99    |
| 12      | 2019-01-02 00:00:00 | 112        | 2        | 200   |
| 12      | 2019-03-31 00:00:00 | 110        | 2        | 299   |

解法：

這題題目看了好幾遍都還看不懂他要衝三小，解出來的值和答案一直不一樣，只好去偷看解答到底這題想問啥邏輯

搞懂後，其實沒有很難但主要要能翻譯並理解題目要問啥。

主要就是購買行為的三個條件，才有達到他認為有 `marketing camping` 的條件

1. 只有在一天內買東西（不符合）
2. 只買過一樣產品類型的東西（不符合）
3. 在第一天買的產品中後來也有買相同產品的（不符合）

去掉上面三個條件後，就是答案，第三個條件其實很難從題目中馬上轉換過來，英文我就爛哭哭🤮🤧



```python
# Import your libraries
import pandas as pd

# Start writing code
marketing_campaign.head()
df = marketing_campaign.copy()
df['date'] = df['created_at'].dt.date

# 只有在一天內買東西
df['purchase_days'] = df.groupby('user_id')['date'].transform('nunique')
# 只買過一樣東西
df['purchase_items'] = df.groupby('user_id')['product_id'].transform('nunique')

# 在第一天買過的東西後來又買
# 每個人在第一天買的東西
df['first_product'] = df.groupby('user_id')['created_at'].transform(lambda x: df.loc[x.idxmin(), 'product_id'])
df['date'] =  pd.to_datetime(df['date'])
df['rank'] = df.groupby('user_id')['date'].rank(method = 'dense')
mask_df = df[df['rank'] == 1]
mask_df['user_product'] = mask_df['product_id'].map(str) + mask_df['user_id'].map(str)


mask = ((df['purchase_days'] > 1) & (df['purchase_items'] > 1))

filtered_df = df[df['user_id'].isin(df.loc[mask, 'user_id'])]
filtered_df['user_product'] = filtered_df['product_id'].map(str) +filtered_df['user_id'].map(str)
filtered_df = filtered_df[~filtered_df['user_product'].isin(mask_df['user_product'])]

len(filtered_df['user_id'].unique())
```

---



### Y-on-Y Growth Rate

公司：Wayfair

難度：Hard

問題核心：子查詢 | window function

問題：

> Assume you are given the table below containing information on user transactions for particular products. Write a query to obtain the [year-on-year growth rate](https://www.fundera.com/blog/year-over-year-growth) for the total spend of each product for each year.
>
> Output the year (in ascending order) partitioned by product id, current year's spend, previous year's spend and year-on-year growth rate (percentage rounded to 2 decimal places).



### `user_transactions`

| transaction_id | product_id | spend   | transaction_date    |
| :------------- | :--------- | :------ | :------------------ |
| 1341           | 123424     | 1500.60 | 12/31/2019 12:00:00 |
| 1423           | 123424     | 1000.20 | 12/31/2020 12:00:00 |
| 1623           | 123424     | 1246.44 | 12/31/2021 12:00:00 |
| 1322           | 123424     | 2145.32 | 12/31/2022 12:00:00 |

解法：

主要要尋找 YoY，一般來說要找 YoY，MoM 這類型的題目都會用到 `LAG()` 或 `LEAD()`，這題也不例外。

1. 首先就是先將 `YEAR` 抽取出來後分組加總
2. 用 LAG() 做出 YoY 的差異然後取變化的百分比

```sql
with base AS(
SELECT a.*, EXTRACT(YEAR FROM transaction_date) AS current_year
FROM user_transactions AS a
),
 
base2 AS (
SELECT current_year, product_id, SUM(spend) AS total_sp
FROM base a
GROUP BY 1, 2
ORDER BY current_year
),

base3 AS (
SELECT a.*, LAG(total_sp) OVER(PARTITION BY product_id ORDER BY current_year) AS win_lag
FROM base2 AS a
)

SELECT 
  a.current_year AS year, 
  a.product_id,
  a.total_sp AS curr_year_spend,
  a.win_lag AS prev_year_spend,
  ROUND(((total_sp - win_lag) / (win_lag))* 100, 2) AS yoy_rate
FROM base3 a

```

---



### Top 5 States With 5 Star Businesses

公司：Yelp

難度：Hard

問題核心：rank | 排序

問題：

> Find the top 5 states with the most 5 star businesses. Output the state name along with the number of 5-star businesses and order records by the number of 5-star businesses in descending order. In case there are ties in the number of businesses, return all the unique states. If two states have the same result, sort them in alphabetical order.



`yelp_business`

| business_id            | name                            | neighborhood | address                   | city      | state | postal_code | latitude | longitude | stars | review_count | is_open | categories                                                   |
| ---------------------- | ------------------------------- | ------------ | ------------------------- | --------- | ----- | ----------- | -------- | --------- | ----- | ------------ | ------- | ------------------------------------------------------------ |
| G5ERFWvPfHy7IDAUYlWL2A | All Colors Mobile Bumper Repair |              | 7137 N 28th Ave           | Phoenix   | AZ    | 85051       | 33.448   | -112.074  | 1     | 4            | 1       | Auto Detailing;Automotive                                    |
| 0jDvRJS-z9zdMgOUXgr6rA | Sunfare                         |              | 811 W Deer Valley Rd      | Phoenix   | AZ    | 85027       | 33.683   | -112.085  | 5     | 27           | 1       | Personal Chefs;Food;Gluten-Free;Food Delivery Services;Event Planning & Services;Restaurants |
| 6HmDqeNNZtHMK0t2glF_gg | Dry Clean Vegas                 | Southeast    | 2550 Windmill Ln, Ste 100 | Las Vegas | NV    | 89123       | 36.042   | -115.118  | 1     | 4            | 1       | Dry Cleaning & Laundry;Laundry Services;Local Services;Dry Cleaning |



解法：

1. 找出 stars 大於五顆星的 business，然後聚合 state count 總數
2.  因為要找 TOP 5，故需要 rank() 排序，這邊題目有說同樣數量的全部都可以算，故不能用 row_number() 和 dens_rank()，其中差異可以 google 比較一下
3. 然後針對數量和 state 排序



`SQL 解法`

```sql
with base as (
    select state, count(distinct business_id) as cmt
    from yelp_business
    where stars >= 5
    group by 1
    order by cmt desc, state
)

select a.state, a.cmt as n_businesses
from (
    select a.*, rank() over(order by cmt desc) as rnk
    from base as a
) as a
where a.rnk <= 5
order by n_businesses desc, state
```



這裡可以提一下 python 在 rank() 函數裡的參數 `method`和 SQL 有比較不一樣的變化， SQL 的 rank() 有在之前的文章提過惹，可參考 [Window Functions in PySpark](https://www.mydatamafia.com/p/window-functions-in-pyspark/)，這邊說明一下 python 的差異

`average`

```python
grouped_df['cmt'].rank(method='average', ascending = False)
```

| state | cmt  | rnk  |
| ----- | ---- | ---- |
| AZ    | 10   | 1    |
| ON    | 5    | 2    |
| NV    | 4    | 3    |
| IL    | 3    | 5    |
| OH    | 3    | 5    |
| WI    | 3    | 5    |
| EDH   | 2    | 7    |
| BW    | 1    | 8.5  |
| QC    | 1    | 8.5  |

會將同名次的元素平均排名，原本為 4, 5, 6，平均則都為第 5 名，8, 9 則平均為 8.5 名

`min`

```python
grouped_df['cmt'].rank(method='average', ascending = False)
```

會將同名次的元素通通給予最前面的排名，原本為 4, 5, 6，平均則都為第 4 名

| state | cmt  | rnk  |
| ----- | ---- | ---- |
| AZ    | 10   | 1    |
| ON    | 5    | 2    |
| NV    | 4    | 3    |
| IL    | 3    | 4    |
| OH    | 3    | 4    |
| WI    | 3    | 4    |
| EDH   | 2    | 7    |
| BW    | 1    | 8    |
| QC    | 1    | 8    |

`max`

就和 min 相反，會返回最後面的排名

`first`

| state | cmt  | rnk  |
| ----- | ---- | ---- |
| AZ    | 10   | 1    |
| ON    | 5    | 2    |
| NV    | 4    | 3    |
| IL    | 3    | 4    |
| OH    | 3    | 5    |
| WI    | 3    | 6    |
| EDH   | 2    | 7    |
| BW    | 1    | 8    |
| QC    | 1    | 9    |

會將同名次的元素按照原本排列的順序排名，類似 SQL 中的 `row_number()`

`desnse`

| state | cmt  | rnk  |
| ----- | ---- | ---- |
| AZ    | 10   | 1    |
| ON    | 5    | 2    |
| NV    | 4    | 3    |
| IL    | 3    | 4    |
| OH    | 3    | 4    |
| WI    | 3    | 4    |
| EDH   | 2    | 5    |
| BW    | 1    | 6    |
| QC    | 1    | 6    |

和 SQL 中的 `DENSE_RANK()`相同，不會跳過相同排名所佔據的排名位置。

python 裡的 rank() 就是將 SQL 中的 `RANK()`再細分為 min, max, average。

`python 解法`

```python
# Import your libraries
import pandas as pd

# Start writing code
yelp_business.head()

df = yelp_business.copy()

df = df.loc[df['stars'] >= 5, :]

grouped_df = df.groupby('state').agg({'business_id': 'nunique'}).reset_index()
grouped_df.columns = ['state', 'cmt']
# 根據 'cmt' 遞減和 'state' 升序進行排序
sorted_df = grouped_df.sort_values(by=['cmt', 'state'], ascending=[False, True])

sorted_df['rnk'] = grouped_df['cmt'].rank(method='min', ascending = False)
top5 = sorted_df.loc[sorted_df['rnk'] <= 5, :].sort_values(by=['cmt', 'state'], ascending=[False, True])
top5[['state', 'cmt']].rename(columns={'cmt': 'n_businesses'}).reset_index(drop=True)
```

---



這一周大概刷了 30-40 題左右，大部分的手感回來滿多的，應該會再找工作的路上每天維持 1-2 題，但好像就要付費了 QQ

