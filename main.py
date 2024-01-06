import pandas as pd
from datetime import datetime
import plotly.graph_objects as ly
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
    def __init__(self, file, ma=10, ema=10, pair = "") -> None:
        self.pair = pair
        
        self.data = pd.read_csv(file, sep="\t")
        
        self.data['<DATETIME>'] = pd.to_datetime(self.data['<DATE>'] + ' ' + self.data['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        self.data = self.data.drop(['<DATE>', '<TIME>', '<VOL>', '<SPREAD>' ], axis=1)
        
        self.data = self.data.rename({"<CLOSE>": "close",
           "<OPEN>": "open",
           "<HIGH>": "high",
           "<LOW>": "low"
           }, axis = 1)
        
        self.data = self.data[['<DATETIME>'] + [col for col in self.data.columns if col != '<DATETIME>']]
        
        self.data['SMA'] = self.data['high'].rolling(window=ma).mean()
        self.data['EMA'] = self.data['high'].ewm(span=ema, adjust=False).mean()
        
        self.chart : list[Candle] = []
        self.n_candles = self.data.shape[0]
        self.start_date = min(self.data["<DATETIME>"])
        self.end_date = max(self.data["<DATETIME>"])
        self.tf = ...
        
        for k in range(self.n_candles):
            c = self.data.loc[k]
            candle = Candle(c["<DATETIME>"], self.tf, c["high"], c["low"], c["open"], c["close"])
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
            
        self.fvg = {"bull":[], "bear": []}
        self.fvg_t = {"bull":[], "bear": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k-1].high < self.chart[k+1].low:
                self.fvg["bull"].append(k)
                self.fvg_t["bull"].append(self.chart[k].timestamp)
                
            if self.chart[k-1].low > self.chart[k+1].high:
                self.fvg["bear"].append(k)
                self.fvg_t["bear"].append(self.chart[k].timestamp)
                
        self.trends = {"bull":[], "bear": []}
        
        trend_type = None
        trend_start = 0
        window = 5
        tolerance = 1
        
        for i in range(self.n_candles-1):
            if trend_type is None:
                if self.chart[i]
            n_bullish, n_bearish = 0, 0
            for c in self.chart[i:i+window]:
                n_bullish += 1 if c.isBullish() else 0
                n_bearish += 1 if c.isBearish() else 0
            
            if trend_type is None:
                if n_bullish > tolerance and n_bearish > tolerance:
                    continue
                elif n_bullish > tolerance:
                    trend_start = i
                    trend_type = "bull"
                elif n_bearish > tolerance:
                    trend_start = i
                    trend_type = "bear"
            
            elif trend_type == "bull":
                if n_bearish > tolerance:
                    

        for i in range(1, self.n_candles):
            if self.chart[i].isBullish():
                if trend_type == "bear":
                    # Bullish trend starts
                    trend_start = i - 1
                trend_type = "bull"
                
            elif self.chart[i].isBearish():
                if trend_type == "bull":
                    # Bearish trend starts
                    trend_start = i - 1
                trend_type = "bear"
            else:
                # Neutral candle, do nothing
                continue

            # Check for trend reversal
            if i < self.n_candles - 1 and trend_type != "neutral":
                if (
                    (trend_type == "bull" and self.chart[i + 1].isBearish()) or
                    (trend_type == "bear" and self.chart[i + 1].isBullish())
                ):
                    # Trend reversal, record the trend
                    self.trends[trend_type].append((trend_start, i))
                    trend_type = None

        # Check for the last trend
        if trend_type:
            self.trends[trend_type].append((trend_start, self.n_candles - 1))


        
            

    def getMSSs(self, return_times = False):
        
        mss = {"bull":[], "bear": []}
        
        for k in range(10,self.n_candles-2):
            lastSwingHigh = self.chart[bisect.bisect_left(self.swi["high"], k)]
            lastSwingLow  = self.chart[bisect.bisect_left(self.swi["low"],  k)]
            
            if self.data["Uptrend"][k] and self.chart[k+1].isBearish() and self.chart[k+2].isBearish() and self.chart[k].close > lastSwingHigh.high:
                mss["bear"].append((self.chart[k].timestamp if return_times else k))
            if self.data["Downtrend"][k] and self.chart[k+1].isBullish() and self.chart[k+2].isBullish() and self.chart[k].close < lastSwingLow.low:
                mss["bull"].append((self.chart[k].timestamp if return_times else k))
                
        return(mss)

    def get_OBs(self, dir):
        OBs = {"bull":[], "bear": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k-1].high < self.chart[k+1].low and (dir is None or dir):
                OBs["bull"].append(self.chart[k].timestamp)
            if self.chart[k-1].low > self.chart[k+1].high and (dir is None or not dir):
                OBs["bear"].append(self.chart[k].timestamp)
        
        return(OBs)
    
    def plotLine(self, start = None, end = None):
        
        start = start if start else self.start_date
        end = end if end else self.end_date
        
        df = self.data[ (self.data["<DATETIME>"] > start) & (self.data["<DATETIME>"] < end) ]
        n = df.shape[0]

        fig = ly.Figure(data=ly.Scatter(x=df['<DATETIME>'],
                            y=df['high'])
                        )
    
        fig.show()
    
    def plot(self, start = None, end = None, BFVG = False, bFVG = False, MSS = False, EMA = False, trends = False):
        
        start = start if start else self.start_date
        end = end if end else self.end_date
        
        df = self.data[ (self.data["<DATETIME>"] >= start) & (self.data["<DATETIME>"] <= end) ]
        n = df.shape[0]

        fig = ly.Figure(data=ly.Candlestick(x=df['<DATETIME>'],
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close']),
                        )
        
        if trends:
            for t in self.trends["bull"] + self.trends["bear"]:
                if t[0] in df.index and t[1] in df.index:
                    fig.add_scatter(x=[df['<DATETIME>'][t[0]], df['<DATETIME>'][t[1]]],
                                    y=[df['open'][t[0]], df['close'][t[1]]],
                                    opacity=0.7,
                                    showlegend=False)
            
        if BFVG:
            for k in self.fvg["bull"]:
                if k in df.index:
                    limit1 = min(k+20, df.index[-1])
                    limit2 = min(k+1, df.index[-1])
                    fig.add_scatter(x=[df['<DATETIME>'][k-1], df['<DATETIME>'][k-1], df['<DATETIME>'][limit1], df['<DATETIME>'][limit1], df['<DATETIME>'][k-1]], 
                                    y=[df['high'][k-1], df['low'][limit2], df['low'][limit2], df['high'][k-1], df['high'][k-1]],
                                    fill="toself",
                                    fillcolor="green",
                                    line=dict(color="green"),
                                    opacity=0.4,
                                    showlegend=False,
                                    )
        
        if bFVG:
            for k in self.fvg["bear"]:
                if k in df.index:
                    limit1 = min(k+20, df.index[-1])
                    limit2 = min(k+1, df.index[-1])
                    fig.add_scatter(x=[df['<DATETIME>'][k-1], df['<DATETIME>'][k-1], df['<DATETIME>'][limit1], df['<DATETIME>'][limit1], df['<DATETIME>'][k-1]], 
                                    y=[df['low'][k-1], df['high'][limit2], df['high'][limit2], df['low'][k-1], df['low'][k-1]],
                                    fill="toself",
                                    fillcolor="red",
                                    line=dict(color="red"),
                                    opacity=0.4,
                                    showlegend=False,
                                    )
        
        # fig.update_layout(yaxis=dict(range=[y_min, y_max])) ...
        
        fig.show()
    

if __name__ == "__main__":
    m = Market("CandleMagic\\data\\EURUSD_M1_231201_231215.csv")
    
    # m.plotLine(start=datetime(2023, 12, 1, 1, 0, 0), end=datetime(2023, 12, 10, 0, 0, 0))
    m.plot(start=datetime(2023, 12, 1, 5, 0, 0), end=datetime(2023, 12, 1, 8, 0, 0), bFVG = False, trends=True)
    
# check https://huggingface.co/dslim/bert-base-NER