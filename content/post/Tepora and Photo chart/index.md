---
author: Pacific2021
categories:
- Camping
- Travel 
date: "2021-12-03"
description: 
image: 
tags:
- Markdown
- Windows
title: Typora + PicGo + Github 圖床設置 
---



## 關於 markdown

---

我自己一向都喜歡使用 markdown 做於學習路上的筆記，在接觸寫程式之前，都用手寫或是 word 來做記錄，而接觸到 markdown 之後著實大為驚豔，簡潔的介面，方便易懂的操作，可嵌入多種工具或是圖片，真的是用過就回不去了。

目前市面上有很多的 markdown 工具，例如 HackMD、dropbox 的 paper 或是給 Mac 用的 MACDORD 等等，而我自己最喜歡用的則是 Typora，他最方便的就是輸入即顯示，不需要分割成 markdown 語法視窗與顯示的畫面，非常的清爽，而且也有很多別人開發的 theme 可以用，聽說 Typora 開源的開發者是強國人，所以很多支援中文字體的 theme 我也覺得是一大優點。

且其支援插入多種格式如數學的 LaTex、表格以及圖片等等，輸出的格式也非常多樣化。

> 很多寫 R 的人會利用 `Blogdown` 的套件直接作為寫作 markdown 的方式，也很方便能直些 render 圖表，不過我還來還是習慣於用 Typora 來寫

這部分的教學可以參考以下文章：

[利用 Blogdown 生成靜態 Blog ](https://www.twblogs.net/a/5c8c8df3bd9eee35cd6ae105)

而我自己後來還是習慣用 Typora 所以我是用 `Typora + Hugo + Github + Netlify` 來生成我的 Blog，方法大同小異，但這就遇到一個問題

`每次 render 圖表都還要存在對應目錄或是手動上傳雲端`，真的麻煩。

因此後來找到可以裡用以下工具來完成自動上傳圖片到雲端圖床的方式，讓圖片管理變簡單，再也不會找不到目錄位置或移動圖而造成無法渲染

> 不過 Typora  2021-12 後進入 1.0 正式版，開始收費了，唯一次性買斷大約 400 台幣，我是當下就 paypal 下去了 :laughing:



## Typora + PicGo + Github 

---

因此透過這三個工具，就能完成 `將圖片拖拉至 Typora 後，圖片會自動上傳到 github 你設定為圖床的 repo`，4 不 4 很方便，

 ### 步驟一：Github 生成 token

> 點擊右上角頭像 > setting > Developer setting

![image-20211229215114110](https://cdn.jsdelivr.net/gh/PacificLin/chart-bed@main/blog/image-20211229215114110.png)

> Developer setting > Personal access tokens > Generate new token

![image-20211229215033960](https://cdn.jsdelivr.net/gh/PacificLin/chart-bed@main/blog/image-20211229215033960.png)

進去後把 `repo` 打勾，Note 可以填提醒自己這個 token ，最後按 `Generate token` 就會生成一串 token，但記得複製下來

因為它就只會顯示這一次。



### 步驟二：安裝 PicGo 和設定

因為我是用 Windows 系統，因此該方法水果的 OS 不適用，水果陣營可以用 uPic 圖床也非常方便，我就窮我買不起 Mac 嗚嗚嗚嗚 :crying_cat_face:。

* 因為安裝 Picgo 需要安裝 nodjs，這邊我是用 scoop 來安裝

```
sccop install nodejs
```

* 然後呢載安裝 npm，npm 為 JS 的套件管理工具

```
sccop install npm
```

* 最後再用 npm 安裝 Picgo 

```
npm install picgo -g
```

* 這時候就可以透過 Picgo 來設定配置

```
picgo set uploader
```

這時候就可以選圖床，我就選 github，網路上應該有其他圖床的配置教學

![image-20211229221754777](https://cdn.jsdelivr.net/gh/PacificLin/chart-bed@main/blog/image-20211229221754777.png)

然後下面的參數設定如下

 **repo：**PacificLin/chart-bed   `Gihub用戶名/ repo`                                                                                                                                                                                                                                                         

**branch：** main                                                                                                                                                                                                                                                                          

**token：** `剛剛步驟一申請的 token`                                                                                                                                                                                                                                       

 **path：** blog/  `上傳在你的 Github repo 跟目錄的資料夾可自己按造需求命名，沒寫就會上傳到跟目錄`                                                                                                                                                                                                                                                                          

**customUrl:** https://cdn.jsdelivr.net/gh/PacificLin/chart-bed@main  `這是 CDN 可以增加圖片 renderder 的速度，可不填`

最後的 CDN 設置可去以下網站研究一下 ：[CDN 連結建立](https://www.jsdelivr.com/?docs=gh)                                                                                                                                                                                                          



### 步驟三：設定 Typora

> Typora > File > Preferances

請將設定如下，因為我們是透過 terminal 來設定，所以選擇這個方法。

`Image Uploader`：Custom Command

`Command`：picgo upload

 <img src="https://cdn.jsdelivr.net/gh/PacificLin/chart-bed@main/blog/image-20211229220755080.png" alt="image-20211229220755080" style="zoom:80%;" />

然後點選 test Uploader，如果成功就會出現以下訊息，好 la 大功告成

<img src="https://cdn.jsdelivr.net/gh/PacificLin/chart-bed@main/blog/image-20211229221254556.png" alt="image-20211229221254556" style="zoom:70%;" />

###  
