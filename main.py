import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

class Candle():
    def __init__(self, t, tf, h, l, o, c) -> None:
        self.timeframe = tf
        self.timestamp = t
        self.high = h
        self.low = l
        self.open = o
        self.close = c
        
        if self.open == self.close:
            self.dir = None
        else:
            self.dir = self.open < self.close # True = Bull, False = Bear, None = None
        
class Market():
    def __init__(self, file) -> None:
        self.data = pd.read_csv(file, sep="\t")
        self.data['<DATETIME>'] = pd.to_datetime(self.data['<DATE>'] + ' ' + self.data['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        self.data = self.data.drop(['<DATE>', '<TIME>', '<VOL>', '<SPREAD>' ], axis=1)
        self.data = self.data[['<DATETIME>'] + [col for col in self.data.columns if col != '<DATETIME>']]
        self.chart : list[Candle] = []
        self.n_candles = self.data.shape[0]
        self.start_date = min(self.data["<DATETIME>"])
        self.end_date = max(self.data["<DATETIME>"])
        self.tf = ...
        
        for k in range(self.n_candles):
            data = self.data.loc[k]
            candle = Candle(data["Date"], self.tf, data["High"], data["Low"], data["Open"], data["Close"])
            self.chart.append(candle)
            
    def get_FVGs(self):
        """
        Returns the ??? of the market's FVGs
        """
        
        FVGs = {"bull":[], "bear": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k-1].high < self.chart[k+1].low:
                FVGs["bull"].append(self.chart[k].timestamp)
            if self.chart[k-1].low > self.chart[k+1].high:
                FVGs["bear"].append(self.chart[k].timestamp)
        
        return(FVGs)

    def get_OBs(self, dir):
        OBs = {"bull":[], "bear": []}
        
        for k in range(1,self.n_candles-1):
            if self.chart[k-1].high < self.chart[k+1].low and (dir is None or dir):
                OBs["bull"].append(self.chart[k].timestamp)
            if self.chart[k-1].low > self.chart[k+1].high and (dir is None or not dir):
                OBs["bear"].append(self.chart[k].timestamp)
        
        return(OBs)
    
    def plot(self, start = None, end = None, FVG = False):
        
        start = start if start else self.start_date
        end = end if end else self.end_date
        
        df = self.data[ (self.data["<DATETIME>"] > start) & (self.data["<DATETIME>"] < end) ]

        fig = go.Figure(data=go.Candlestick(x=df['<DATETIME>'],
                            open=df['<OPEN>'],
                            high=df['<HIGH>'],
                            low=df['<LOW>'],
                            close=df['<CLOSE>']))
        
        if FVG:
            fig.add_scatter(...)
        
        fig.show()
    

if __name__ == "__main__":
    m = Market("CandleMagic\\data\\EURUSD_M1_231201_231215.csv")