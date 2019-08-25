# high_freq_trading

In this project, I analyse high frequency trading data. I am grateful for all the help I have received from everyon who has taught and guided me on this subject.

High frequency trading refers to trading in short time periods and using algorithmic strategies to capture alpha. Unfortunately these alphas decay extremely quickly due to overcrowding and seeking new methods of capturing trends is always necessary. 

The data I have used is unfortunately private and confidential and cannot be uploaded onto this platform.
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

In the first step, we calculate the smart price and future smart price of every row. The smart price consists of (BidPrice1 x AskVol1 + AskPrice1 x BidVol1)/(AskVol1+BidVol1) and future smart price refers to the smart price 30 seconds from the current price. Edge is then calculated (future smart price - current smart price)
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

We then (linearly) regress this value against edge for a 5 day rolling window. We also run regressions against all 4 moving average times. One such result can be found at:
https://github.com/huddyyeo/high_freq_trading/blob/master/result_5day_15minute_ma.csv
We interpret the data as, for the 5 day period ending in 2019.01.08, the regression of edge against against (smart price - SMA) returns a slope of 0.0157977748172754 and p-value of the coefficient of 2.0954766362309706e-93.
Note that the p-values are all extremely small due to the sheer size of the dataset. The slope is also inconsistent, flipping between positive and negative, meaning we cannot consistently tell how being on either side of the SMA affects the edge and future price. 

Now, I hypothesize that large values of (price-SMA) lead to mean reversion, while small values just indicate momentum. 

Next step, we split the (smartprice-SMA) into positive and negative. Within positive and negative, identify each category's Q1,Q2,Q3 and Q4 over a 20 day rolling window. The categories are defined as such:

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

Note that limit up events occur, which result in the dataset having a 0 ask price. The smart-price calculation function is updated as such:

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

The data is returned in the following format

| Date        | category_1_slope |
| ------------| ---------------- |
| 2019.01.30  | -0.09            |

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
In category 8, (price-MA) is very positive. A negative slope means, edge is negative, indicating a price much higher than the moving average leads to a future decrease in price.
Thus we observe that a negative slope in category 1 and 8 makes sense and it indicates mean-reversion.
A positive slope means that a positive value of (price-MA) leads to a future increase in price and vice verse for negative values. This is hence momentum and can be expected for smaller values of (price-MA). We have thus identified a pattern in the trading data.
