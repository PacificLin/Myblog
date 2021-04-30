---
author: Pacific2021
categories:
- Analysis
- R
date: "2021-04-20"
description: R 語言利用 ave() & cunsum() 做滾動計數的方法
image: 
tags:
- R
- Data Manipulating
- Analysis
title: 資料處理－連續計數
---

這次公司的 APP 有新產品，而新產品有某個功能用戶可以隨意調整功能要打開還是關閉，而後端系統會在某個月的特定時間去抓取用戶該功能是開啟還是關閉，

如果功能開啟 `open_flag == 1`

如果功能關閉 `open_flag == 0`

以下我建立假資料來模擬 data，模擬 10 個月的開關紀錄，這次是使用 R 語言來做處理

```R
name <- c("Ann", "Teddy", "Frank", "Evie")
sample(name, replace = TRUE, size = 20)
rbinom(20, 1, 0.7)

df <- data.frame(name = rep(name, each = 10),
                 open_flag = rbinom(40, 1, 0.7),
                 date = rep(seq(as.Date('2020/01/01'), by = 'month', length.out = 10), times = 4, each = 1))
```

```R
   name open_flag       date
1    Ann         0 2020-01-01
2    Ann         1 2020-02-01
3    Ann         1 2020-03-01
4    Ann         1 2020-04-01
5    Ann         1 2020-05-01
6    Ann         1 2020-06-01
7    Ann         1 2020-07-01
8    Ann         1 2020-08-01
9    Ann         0 2020-09-01
10   Ann         1 2020-10-01
11 Teddy         1 2020-01-01
12 Teddy         0 2020-02-01
13 Teddy         0 2020-03-01
14 Teddy         1 2020-04-01
15 Teddy         1 2020-05-01
16 Teddy         1 2020-06-01
17 Teddy         1 2020-07-01
18 Teddy         0 2020-08-01
19 Teddy         0 2020-09-01
20 Teddy         0 2020-10-01
21 Frank         0 2020-01-01
22 Frank         1 2020-02-01
23 Frank         1 2020-03-01
24 Frank         1 2020-04-01
25 Frank         0 2020-05-01

```

處理這個類型 table 的需求花了我一點時間，因此記錄一下做法，這個 table 最後要處理的樣態為：統計每個客戶連續開啟該功能最久的時間，簡單來說找出最多有連續幾個 1，

假設客戶可能開開關關，資料樣態如下

```R
(1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1)
```

則客戶連續開啟該功能最久的時間是 4 個時間段，做多有 4 個連續的 1

> 思考邏輯：也就是遇到 0 後 open_flag 要重新加總

一開始我想用 `lag()` 來解，但後來發現用 lag 只能找出`開 -> 關` 在哪一個 row，但沒有辦法加總，因此最重要的點還是在於如何 `遇到 0 後可以重新加總`，因此應該是要用 `cunsum()` 做為解法基礎，來計算連續的時間區段

#### 步驟一

---

先依照客戶姓名將 data.frame 轉換成 list

```R
df_ls <- split(df, f = df$name)
```

#### 步驟二

---

利用 `lapply()` 將 `ave()` 函數做分組計算並 `cumsum()` 加總

```r
df_ls <- lapply(df_ls, function(x) {
  mutate(x, cum_pd = ave(open_flag, cumsum(open_flag == 0), FUN = cumsum))
})
```

```R
$Ann
   name open_flag       date cum_pd
1   Ann         0 2020-01-01      0
2   Ann         1 2020-02-01      1
3   Ann         1 2020-03-01      2
4   Ann         1 2020-04-01      3
5   Ann         1 2020-05-01      4
6   Ann         1 2020-06-01      5
7   Ann         1 2020-07-01      6
8   Ann         1 2020-08-01      7
9   Ann         0 2020-09-01      0
10  Ann         1 2020-10-01      1

$Evie
   name open_flag       date cum_pd
1  Evie         0 2020-01-01      0
2  Evie         0 2020-02-01      0
3  Evie         1 2020-03-01      1
4  Evie         1 2020-04-01      2
5  Evie         1 2020-05-01      3
6  Evie         1 2020-06-01      4
7  Evie         0 2020-07-01      0
8  Evie         1 2020-08-01      1
9  Evie         1 2020-09-01      2
10 Evie         1 2020-10-01      3

$Frank
    name open_flag       date cum_pd
1  Frank         0 2020-01-01      0
2  Frank         1 2020-02-01      1
3  Frank         1 2020-03-01      2
4  Frank         1 2020-04-01      3
5  Frank         0 2020-05-01      0
6  Frank         1 2020-06-01      1
7  Frank         1 2020-07-01      2
8  Frank         1 2020-08-01      3
9  Frank         1 2020-09-01      4
10 Frank         1 2020-10-01      5
```

從上面結果就可以看的出來，遇到 0 也就是關閉功能後，連續的打開的時間段記數就會重新開始計算

#### 步驟三

---

利用 `do.call` 合併回 data.frame 最後找出最大值

```R
do.call(rbind, df_ls) %>%
  group_by(name) %>% 
  summarize(max_flag = max(cum_pd))
```

```R
  name  max_flag
  <fct>    <int>
1 Ann          7
2 Evie         4
3 Frank        5
4 Teddy        4
```

結果就會如上，可以找出用戶開啟該功能的最大時間段

而整個資料處理過程有幾個重要的 function 可以好好整理如下





[r - R中的ave（）函数和mean（）函数有什么区别？ - 堆栈内存溢出 (stackoom.com)](https://stackoom.com/question/3t6uV/R中的ave-函数和mean-函数有什么区别)

[r - Cumulative sum that resets when 0 is encountered - Stack Overflow](https://stackoverflow.com/questions/32501902/cumulative-sum-that-resets-when-0-is-encountered)