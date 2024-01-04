import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import bisect

class Candle():
    def __init__(self, t, tf, h, l, o, c) -> None:
        self.timeframe = tf
        self.timestamp = t
        self.high = h
        self.low = l
        self.open = o
        self.close = c
            
    def isBullish(self):
        return(self.open > self.close)
    
    def isBearish(self):
        return(self.open < self.close)

    def bodySize(self):
        return(abs(self.open-self.close))
        
class Market():
    def __init__(self, file, ma=10, ema=10) -> None:
        self.data = pd.read_csv(file, sep="\t")
        
        self.data['<DATETIME>'] = pd.to_datetime(self.data['<DATE>'] + ' ' + self.data['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        self.data = self.data.drop(['<DATE>', '<TIME>', '<VOL>', '<SPREAD>' ], axis=1)
        self.data = self.data[['<DATETIME>'] + [col for col in self.data.columns if col != '<DATETIME>']]
        
        self.data['SMA'] = self.data['Price'].rolling(window=ma).mean()
        self.data['EMA'] = self.data['Price'].ewm(span=ema, adjust=False).mean()
        
        self.chart : list[Candle] = []
        self.n_candles = self.data.shape[0]
        self.start_date = min(self.data["<DATETIME>"])
        self.end_date = max(self.data["<DATETIME>"])
        self.tf = ...
        
        for k in range(self.n_candles):
            data = self.data.loc[k]
            candle = Candle(data["Date"], self.tf, data["High"], data["Low"], data["Open"], data["Close"])
            self.chart.append(candle)
            
        uptrends =   ma*[np.nan] + [ (sum([ int(self.chart[i].isBullish()) for i in range(k-ma, k) ]) > int(ma*0.75)) for k in range(ma, self.n_candles) ]
        downtrends = ma*[np.nan] + [ (sum([ int(self.chart[i].isBearish()) for i in range(k-ma, k) ]) > int(ma*0.75)) for k in range(ma, self.n_candles) ]
        
        self.data["Uptrend"] = pd.Series(uptrends)
        self.data["Downtrend"] = pd.Series(downtrends)
        
        self.swi = {"high":[], "low": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k].high > self.chart[k-1].high and self.chart[k].high > self.chart[k+1].high:
                self.swi["high"].append(k)
            if self.chart[k].low < self.chart[k-1].low and self.chart[k].low < self.chart[k+1].low:
                self.swi["low"].append(k)
            
    def getFVGs(self, return_times = False):
        """
        Returns the ??? of the market's FVGs
        """
        
        fvg = {"bull":[], "bear": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k-1].high < self.chart[k+1].low:
                fvg["bull"].append((self.chart[k].timestamp if return_times else k))
                
            if self.chart[k-1].low > self.chart[k+1].high:
                fvg["bear"].append((self.chart[k].timestamp if return_times else k))
        
        return(fvg)
    
    def getMSSs(self, return_times = False):
        
        mss = {"bull":[], "bear": []}
        
        for k in range(10,self.n_candles-2):
            lastSwingHigh = self.chart[bisect.bisect_left(self.swi["high"], k)]
            lastSwingLow  = self.chart[bisect.bisect_left(self.swi["low"],  k)]
            
            if self.data["Uptrend"][k] and self.chart[k+1].isBearish() and self.chart[k+2].isBearish() and self.chart[k].close > lastSwingHigh.high:
                mss["bear"].append((self.chart[k].timestamp if return_times else k))
            if self.data["Downtrend"][k] and self.chart[k+1].isBullish() and self.chart[k+2].isBullish() and self.chart[k].close < lastSwingLow.low:
                mss["bull"].append((self.chart[k].timestamp if return_times else k))
                
        

    def get_OBs(self, dir):
        OBs = {"bull":[], "bear": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k-1].high < self.chart[k+1].low and (dir is None or dir):
                OBs["bull"].append(self.chart[k].timestamp)
            if self.chart[k-1].low > self.chart[k+1].high and (dir is None or not dir):
                OBs["bear"].append(self.chart[k].timestamp)
        
        return(OBs)
    
    def plot(self, start = None, end = None, FVG = False, MSS = False, EMA = False):
        
        start = start if start else self.start_date
        end = end if end else self.end_date
        
        df = self.data[ (self.data["<DATETIME>"] > start) & (self.data["<DATETIME>"] < end) ]

        fig = go.Figure(data=go.Candlestick(x=df['<DATETIME>'],
                            open=df['<OPEN>'],
                            high=df['<HIGH>'],
                            low=df['<LOW>'],
                            close=df['<CLOSE>']))
        
        fig.add
        
        if FVG:
            fig.add_scatter(...)
        
        fig.show()
    

if __name__ == "__main__":
    m = Market("CandleMagic\\data\\EURUSD_M1_231201_231215.csv")