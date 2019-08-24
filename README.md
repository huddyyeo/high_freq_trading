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

Next step, we split the (smartprice-SMA) into positive and negative and identify each category's Q1,Q2,Q3 and Q4 over a 20 day rolling window. We then run regressions on these 8 categories to identify the trends. We would expect mean reversion to occur in the extreme quartiles. To be continued.





