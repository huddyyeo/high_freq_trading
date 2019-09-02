## high_freq_trading 

# work in progress

In this project, I analyse high frequency trading data. I am grateful for all the help I have received on this subject.

High frequency trading refers to trading in short time periods and using algorithmic strategies to capture alpha. Unfortunately, these alphas decay extremely quickly due to overcrowding and seeking new methods of capturing trends is always necessary. 

The data I have used is unfortunately confidential and cannot be shared here.
It consists of the following columns:
['date', 'ReceiveTime', 'Symbol', 'Type', 'PreSettlePrice',
       'PreClosePrice', 'PreOpenInterest', 'OpenPrice', 'HighestPrice',
       'LowestPrice', 'LastPrice', 'BidPrice1', 'BidPrice2', 'BidPrice3',
       'BidPrice4', 'BidPrice5', 'AskPrice1', 'AskPrice2', 'AskPrice3',
       'AskPrice4', 'AskPrice5', 'BidVol1', 'BidVol2', 'BidVol3', 'BidVol4',
       'BidVol5', 'AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
       'Volume', 'Turnover', 'OpenInterest', 'ClosePrice', 'SettlePrice',
       'WAvgPrice', 'HistoricalHigh', 'HistoricalLow', 'UpperLimitPrice',
       'LowerLimitPrice', 'TotalBidVol', 'TotalAskVol', 'InternalDate', 'Time',
       'CalendarDate', 'Category', 'InternalTime', 'TradingDate',
       'BidWAvgPrice', 'AskWAvgPrice']

In the first step, we calculate the smart price and future smart price of every row. The smart price consists of (BidPrice1 x AskVol1 + AskPrice1 x BidVol1)/(AskVol1+BidVol1).
Note that limit up events occur, which result in the dataset having a 0 ask price. The smart-price calculation function is as such:

```
def calc_smart_price(dataset):
    data=dataset[:]
    
    #to combat the limit up event, where price is set to 0. 
    rows=(data.loc[:,'BidPrice1']==0) #count rows of bid price equal 0
    if (np.any(rows)): #if there is such a row
        data.at[rows,'BidPrice1']=data.loc[rows,'AskPrice1'] #for that row, assign ask price to it
    rows=(data.loc[:,'AskPrice1']==0) #do the same for ask price
    if (np.any(rows)):
        data.at[rows,'AskPrice1']=data.loc[rows,'BidPrice1'] 
        
    data['smart_price']=data.loc[:,'BidPrice1']*data.loc[:,'AskVol1']+data.loc[:,'AskPrice1']*data.loc[:,'BidVol1']
    data.at[:,'smart_price']=data.loc[:,'smart_price']/(data.loc[:,['BidVol1','AskVol1']].sum(axis=1))  
    return data
```

Future smart price refers to the smart price 30 seconds from the current price. Edge is then calculated (future smart price - current smart price)
The simple moving average is then calculated of the current smart price. Since the data is extremely large(>1gb), an efficient way to calculate SMA is necessary. 

```
def calc_sma_fast(dataset,duration=1):
    data=dataset[:]
    sma_values=[] 
    smart_sum=np.cumsum(data[:,51])
    for i in range(len(data)):
        last_time=data[i,44]-timedelta(minutes=duration)
        j=220*duration#4x60=240
        while(i-j>0 and data[i-j,44]>last_time):
            j+=1
        if (i-j>=0):
            sma=(smart_sum[i]-smart_sum[i-j])/(j)
            sma_values.append(sma)
        else:
            sma=smart_sum[i]/(i+1)
            sma_values.append(sma)
    sma_values=np.asarray(sma_values)
    sma_values=data[:,51]-sma_values
    sma_values=np.expand_dims(sma_values,axis=1)
    return np.concatenate((data,sma_values),axis=1)     
```
This method returns in 0.05s for 5000 rows, the fastest out of several methods I have tested.
The added column is (smart_price-SMA). The function is then run for 1,5,15,30 minute moving averages.

We then (linearly) regress this value against edge for a 5 day rolling window. Note that we do not fit the intercept (force line through (0,0)). The following code is used
```
class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                     n_jobs=1):
            self.fit_intercept = fit_intercept
            self.normalize = normalize
            self.copy_X = copy_X
            self.n_jobs = n_jobs
    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self  
```
This builds on the sklearn linear_model class and adds the functionality of retrieving p values.

We also run regressions against all 4 moving average times. One such result can be found at:
https://github.com/huddyyeo/high_freq_trading/blob/master/result_5day_15minute_ma.csv
We interpret the data as, for the 5 day period ending in 2019.01.08, the regression of edge against against (smart price - SMA) returns a slope of 0.0157977748172754 and p-value of the coefficient of 2.0954766362309706e-93.
Note that the p-values are all extremely small due to the sheer size of the dataset. The slope is also inconsistent, flipping between positive and negative, meaning we cannot consistently tell how being on either side of the SMA affects the edge and future price. 

Now, I hypothesize that large values of (price-SMA) lead to mean reversion, while small values just indicate momentum. 

Next step, we split the (smartprice-SMA) into positive and negative values. Within positive and negative, identify each category's Q1,Q2,Q3 and Q4 over a 20 day rolling window. The categories are defined as such:

Category:
1: negative 0-25%
2: negative 25-50%
3: negative 50-75%
4: negative 75-100%
5: positive 0-25%
6: positive 25-50%
7: positive 50-75%
8: positive 75-100%

We then run linear regressions within these 8 categories of edge against (price-SMA). We would expect mean reversion to occur in the extreme quartiles. 


The data is returned in the following format

| Date        | category_1_slope |category_2_slope|category_3_slope|category_4_slope|
| ------------| ---------------- |----------------|----------------|----------------|
| 2019.01.30  | -0.09            | xxx            | xxx            | xxx            |

with all 8 quartiles and other dates. It is interpreted as: for the 20 day period ending in 2019.01.30, the data points of 2019.01.30 that have been allocated into category 1 have been regressed and have a slope of -0.09. The p-values of all the rows are small due to large number of observations.

Next, we analyse this result and come to the following conclusion.

| Category | Percentage days positive | Mean slope | Std dev of slope |
| -------- | ------------------------ | ---------- | ---------------- |
| 1        |    0.3650794             |  -0.0944691|  0.2561066       |
| 2        |    0.4603175             |  0.0002484 |  0.577241        |
| 3        |    0.5873016             |  0.0562178 |  0.7733827       |
| 4        |    0.6825397             |  0.3807183 |  0.8314094       |
| 5        |    0.6507937             |  0.1983677 |  0.8524453       |
| 6        |    0.5714286             |  0.0059226 |  0.6174626       |
| 7        |    0.5238095             |  -0.0049956|  0.472878        |
| 8        |    0.3968254             |  -0.0651061|  0.2246008       |

We see somewhat of a trend, where the extreme quartiles are more likely to have a negative slope.
In category 1, (price-MA) is very negative. A negative slope means edge is positive, indicating a price much lower than the moving average leads to a future increase in price.
In category 8, (price-MA) is very positive. A negative slope means edge is negative, indicating a price much higher than the moving average leads to a future decrease in price.
Thus we observe that a negative slope in category 1 and 8 makes sense and it indicates mean-reversion.
A positive slope means that a positive value of (price-MA) leads to a future increase in price and vice verse for negative values. This is hence momentum and can be expected for smaller values of (price-MA). We have successfully identified a pattern in the trading data. 
code: 1day_8split_MA_regression.ipynb

We now look at the effect of volume. We theorise that volume affects how (price-SMA) affects edge. We split volume into 6 categories based on the 5 percentiles (10,25,50,75,90) thresholds calculated from the past 20 days' volumes. 25, 50 and 75 was originally tested, but the 25% percentile proved too high and did not accurately capture the effect of very low volumes.
Similarly, we split the rows into 6 categories and regress (price-SMA) and edge.

We obtain the follow result:

| Category | Percentage days positive| Mean slope | Std dev of slope |
| -------- | ----------------------- | ---------- | ---------------- |
| 1        |    0.515625             |  0.0354863 |  0.2617837       |
| 2        |    0.687500             |  0.0235138 |  0.0772954       |
| 3        |    0.531250             |  0.0206701 |  0.0509659       |
| 4        |    0.562500             |  0.0137058 |  0.0622627       |
| 5        |    0.406250             | -0.0098534 |  0.0591091       |
| 6        |    0.390625             | -0.0485320 |  0.1117760       |

The results make sense. The result say that a low volume in Category 1 does not conclusively tell us whether momentum or mean reversion is occurring. Conversely, a very high volume shows that mean reversion is occurring.
code: 1day_Volume_6split_MA_regression.ipynb

Combining both signals, we split the data into 6 categories based on volume and 8 categories based on SMA and regress the 6x8=42 categories. code: 1day_Q1,8_SMA_vol6split.ipynb

However on looking through scatterplots of the data, we realised that there exist multiple relationships within (price-SMA) and edge. It would make better sense to first evaluate them without regressions. First we split the data into 20 categories (10-90% percentile of positive and negative, 10x2) based on past 20 days of (price-SMA). Then within each category from the past 20 days, we further split into 4 quartiles, based on cumulative volume in the past 1 minute. Within each quartile, we calculate the mean of the edge for all days.

One such result is:

### Mean edge across all days per quartile

|Category|Q1     |Q2      |Q3      |Q4      |
|--------|-------|--------|--------|--------|
|1       |0.14   |   0.   |  0.174 |  0.721 |
|2       |0.065  | -0.117 |  0.047 | -0.073 |
|3       |0.039  |  0.103 | -0.05  |  -0.116|
|4       |0.013  | -0.122 | -0.038 | -0.026 |
|5       |-0.152 | -0.255 | -0.242 | -0.2   |
|6       |-0.312 | -0.183 | -0.233 | -0.156 |
|7       | -0.228| -0.203 | -0.222 |  0.002 |
|8       | -0.267| -0.245 | -0.125 | -0.036 |
|9       | -0.196| -0.157 | -0.133 | -0.044 |
|10      | -0.165| -0.104 | -0.018 | -0.009 |
|11      | -0.017| -0.016 | -0.089 | -0.109 |
|12      |0.072  |  0.143 |  0.079 |  0.034 |
|13      |-0.037 |  0.168 |  0.175 | -0.013 |
|14      |0.102  |  0.09  |   0.077| -0.03  |
|15      |0.09   | 0.346  |  0.016 |  0.03  |
|16      |0.185  |  0.23  |   0.111|  0.083 |
|17      |0.165  |  0.216 |  0.298 |  0.023 |
|18      |0.128  |  0.294 |  0.169 |  0.116 |
|19      |0.146  |  0.029 |  0.305 |  0.107 |
|20      |0.06   |   0.068| -0.155 | -0.583 |
 
Each row refers to a category, 1-20 from top to bottom, and each column is a quartile, Q1-4 left to right. We see that the first 10 rows of mean edge are more likely to be negative, which means momentum, since category 1-10 indicates negative (price-SMA). In the more extreme categories such as the first row, we see that the mean edge is positive. This indicates mean-reversion and it occurs when price is extremely below moving average (bottom 10%). Conversely, we see the opposite in the bottom half of the matrix. Highly positive categories lead to negative mean edge and hence mean reversion. Mean reversion is especially pertinent in quartile 4, or the right-most column. This is expected: when volume is high and price is on the extreme on either side of SMA, we would expect a mean reversion. 
 
code: auto_runall_10split_cat1_split_6_mean.ipynb
The code also contains functions to run the analysis on other products in other folders.

To be continued.
