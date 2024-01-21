import pandas as pd
import datetime
import plotly.graph_objects as ly
import matplotlib.pyplot as plt
import numpy as np
import bisect
from typing import *
import os
import shutil

DAY_NAMES={0: "Monday",
           1: "Tuesday",
           2: "Wednesday",
           3: "Thursday",
           4: "Friday"}

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
    def __init__(self, data: Union[str, pd.DataFrame], ma=10, ema=10, pair = "", mode = "") -> None:
        
        self.mode = mode
        
        self.ma = ma
        self.ema = ema
        
        if isinstance(data, str):
            self.data = pd.read_csv(data, sep="\t")
            self.pair = data.split("\\")[-1].split("/")[-1].split("_")[0]
            self.data['ts'] = pd.to_datetime(self.data['<DATE>'] + ' ' + self.data['<TIME>'], format='%Y.%m.%d %H:%M:%S')
            self.data['ts'] = self.data['ts'].map(lambda x: x - datetime.timedelta(hours = 7)) # NY time
            
            self.data = self.data.drop([ '<DATE>', '<TIME>', '<VOL>', '<SPREAD>' ], axis=1)
            
            self.data = self.data.rename({"<CLOSE>": "close",
                                        "<OPEN>": "open",
                                        "<HIGH>": "high",
                                        "<LOW>": "low"
                                            }, axis = 1)
            
            self.data = self.data[['ts'] + [col for col in self.data.columns if col != 'ts']]
            
            self.data['SMA'] = self.data['high'].rolling(window=ma).mean()
            self.data['EMA'] = self.data['high'].ewm(span=ema, adjust=False).mean()
            
        else: # The input has already been formatted
            self.data = data
            self.pair = pair
        
        self.chart : list[Candle] = []
        self.n_candles = self.data.shape[0]
        self.start_date = min(self.data["ts"])
        self.end_date = max(self.data["ts"])
        self.tf = ...
        
        for k in range(self.n_candles):
            c = self.data.loc[k]
            candle = Candle(c["ts"], self.tf, c["high"], c["low"], c["open"], c["close"])
            self.chart.append(candle)
            
        self.uptrends =   ma*[np.nan] + [ sum([ int(self.chart[i].isBullish()) for i in range(k-ma, k) ]) > int(ma*0.75) for k in range(ma, self.n_candles) ]
        self.downtrends = ma*[np.nan] + [ sum([ int(self.chart[i].isBearish()) for i in range(k-ma, k) ]) > int(ma*0.75) for k in range(ma, self.n_candles) ]
        
        self.data["Uptrend"] = pd.Series(self.uptrends)
        self.data["Downtrend"] = pd.Series(self.downtrends)
        
        self.swi = {"high":[], "low": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k].high > self.chart[k-1].high and self.chart[k].high > self.chart[k+1].high:
                self.swi["high"].append(k)
            if self.chart[k].low < self.chart[k-1].low and self.chart[k].low < self.chart[k+1].low:
                self.swi["low"].append(k)
        
        
        ##### FVGs and price legs #####
        
        self.fvg = {"bull": [], 
                    "bear": []}
        
        self.price_legs = {"bull": {}, 
                           "bear": {}}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k-1].high < self.chart[k+1].low:
                self.fvg["bull"].append(k)
                
                leg_low = self.chart[k-1].low
                leg_low_body = min(self.chart[k-1].close, self.chart[k-1].open)
                for c in range(k+1, len(self.chart)):   # Looking at next candles until we find a bearish one, ending the bullish move
                    if self.chart[c].isBearish():       # End of price move
                        leg_high = max(self.chart[c-1].high, self.chart[c].high)
                        leg_high_body = self.chart[c-1].close
                        break
                self.price_legs["bull"][k] = [[k, c], [leg_low, leg_high], [leg_low_body, leg_high_body]]
                
                if len(self.price_legs["bull"]) > 1 and self.price_legs["bull"][k][0][1] == self.price_legs["bull"][self.fvg["bull"][-2]][0][1]: # The FVG is part of the price leg of the previous FVG 
                    self.price_legs["bull"][k] == self.price_legs["bull"][self.fvg["bull"][-2]]                                                  # because their price legs end at the same candle
                
                # This adjustment propagates from FVG to FVG within the same price leg
                
            if self.chart[k-1].low > self.chart[k+1].high:
                self.fvg["bear"].append(k)
                
                leg_high = self.chart[k-1].high
                leg_high_body = max(self.chart[k-1].close, self.chart[k-1].open)
                for c in range(k+1, len(self.chart)):
                    if self.chart[c].isBullish():
                        leg_low = min(self.chart[c-1].low, self.chart[c].low)
                        leg_low_body = self.chart[c-1].close
                        break
                    
                self.price_legs["bear"][k] = [[k, c], [leg_high, leg_low], [leg_high_body, leg_low_body]]
                
                if len(self.price_legs["bear"]) > 1 and self.price_legs["bear"][k][0][1] == self.price_legs["bear"][self.fvg["bear"][-2]][0][1]:
                    self.price_legs["bear"][k] == self.price_legs["bear"][self.fvg["bear"][-2]]
                    
        
        self.OBs = {"bull":[], "bear": []}
        
        for k in range(self.n_candles-3):
            if self.chart[k].isBullish() and self.chart[k+1].isBearish() and self.chart[k+2].isBearish() and self.chart[k+3].isBearish():
                self.OBs["bull"].append(k)
            if self.chart[k].isBearish() and self.chart[k+1].isBullish() and self.chart[k+2].isBullish() and self.chart[k+3].isBullish():
                self.OBs["bear"].append(k)
        
        """
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
        """

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
    
    def plotLine(self, start = None, end = None):
        
        start = start if start else self.start_date
        end = end if end else self.end_date
        
        df = self.data[ (self.data["ts"] > start) & (self.data["ts"] < end) ]
        n = df.shape[0]

        fig = ly.Figure(data=ly.Scatter(x=df['ts'],
                            y=df['high'])
                        )
    
        fig.show()
    
    def plot_NYsessions(self, start = None, end = None):
        days = list(self.data["ts"].map(lambda x: str(x)[:10]).unique())
        dirname = f"imgs_{datetime.datetime.now().__format__('%Y%m%d_%H')}"
        try:
            shutil.rmtree(dirname)
        except FileNotFoundError:
            pass
        os.mkdir(dirname)
        
        for day in days:
            day_L = day.split("-")
            startNY, endNY = datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 7, 30, 0), datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 12, 0, 0)
            if startNY < start or startNY.weekday() > 4:
                continue
            
            if endNY > end:
                break
            
            df = self.data[ (self.data["ts"] >= startNY) & (self.data["ts"] <= endNY) ]
            fig = ly.Figure(data=ly.Candlestick(x=df['ts'],
                                                open=df['open'],
                                                high=df['high'],
                                                low=df['low'],
                                                close=df['close']),
                        )
            
            fig.write_image(f"{dirname}/{self.pair}_{day.replace('-', '')}_NY.png")
        
        fig = ly.Figure()
        
        cnt = 0
        for pic in os.listdir(dirname):
            fig.add_layout_image(
                source=f"{dirname}/{pic}",
                x=cnt%5,
                y=cnt//5,
                xanchor="center",
                yanchor="middle",
            )
            
            cnt += 1

        # Update layout settings
        # fig.update_layout(
        #     images=[
        #         dict(
        #             source="path/to/image3.png",
        #             x=0.5,
        #             y=0.5,
        #             xanchor="center",
        #             yanchor="middle",
        #         )
        #     ]
        # )

        # Show the figure
        fig.show()
        

    def plot(self, start = None, end = None, BFVG = False, bFVG = False, MSS = False, EMA = False, trends = False):
        
        start = start if start else self.start_date
        end = end if end else self.end_date
        
        df = self.data[ (self.data["ts"] >= start) & (self.data["ts"] <= end) ]
        n = df.shape[0]

        fig = ly.Figure(data=ly.Candlestick(x=df['ts'],
                                            open=df['open'],
                                            high=df['high'],
                                            low=df['low'],
                                            close=df['close']),
                        )
        
        mini = min(df["low"])
        # maxi = max(self.data["high"])
        
        if trends:
            for i in df.index:
                if self.data["Uptrend"][i]:
                    fig.add_scatter(x=[df['ts'][i]],
                                    y=[mini],
                                    opacity=1,
                                    fillcolor="green",
                                    line=dict(color="green"),
                                    showlegend=False)
                    
                if self.data["Downtrend"][i]:
                    fig.add_scatter(x=[df['ts'][i]],
                                    y=[mini],
                                    opacity=1,
                                    fillcolor="red",
                                    line=dict(color="red"),
                                    showlegend=False)
        
        """
        if trends:
            for t in self.trends["bull"] + self.trends["bear"]:
                if t[0] in df.index and t[1] in df.index:
                    fig.add_scatter(x=[df['ts'][t[0]], df['ts'][t[1]]],
                                    y=[df['open'][t[0]], df['close'][t[1]]],
                                    opacity=0.7,
                                    showlegend=False)
        """
        
        if BFVG:
            for k in self.fvg["bull"]:
                if k in df.index:
                    limit1 = min(k+20, df.index[-1])
                    limit2 = min(k+1, df.index[-1])
                    fig.add_scatter(x=[df['ts'][k-1], df['ts'][k-1], df['ts'][limit1], df['ts'][limit1], df['ts'][k-1]], 
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
                    fig.add_scatter(x=[df['ts'][k-1], df['ts'][k-1], df['ts'][limit1], df['ts'][limit1], df['ts'][k-1]], 
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
    os.chdir("CandleMagic")
    m = Market("data/EURUSD_M1_231201_231215.csv")
    m.plot_NYsessions(start=datetime.datetime(2023, 12, 2, 0, 0, 0), end=datetime.datetime(2023, 12, 15, 23, 59, 59))