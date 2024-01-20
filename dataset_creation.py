from classes import Candle, Market
import pandas as pd
import datetime
import plotly.graph_objects as ly
import matplotlib.pyplot as plt
import numpy as np
import bisect
import argparse

# c:/Users/adame/OneDrive/Bureau/CODE/.venv/Scripts/python.exe c:/Users/adame/OneDrive/Bureau/CODE/CandleMagic/dataset_creation.py -f .\CandleMagic\data\EURUSD_M1_231201_231215.csv

"""
    Name	Description
    DAY     Day of week
RET_BFVG	Price retraced into a Bullish FVG
RET_BFVG_D	Price retraced into a Bullish FVG in a Discount
LIQ_LTH_X	Took long-term high (sell-side liquidity) (previous high/TBL)
BFVG_VLTN	Price closed below (violated) a Bullish FVG
BTREND_D	Daily bullish trend (past 3 days closed above open)
TRCSLD_P30M	A tight-range consolidation happened within the past 30 Minutes
DAYOP_ABV	Number of 1M candles closing above the day opening (00:00 AM)
SESOP_ABV	Number of 1M candles closing above the NY session opening (7:30 AM)

Target variable : 0 = No significant move, 1 = Bullish move, 2 = Bearish move
"""

BIG_MOVE = 35 # pips
HISTORY_DAYS = 10 # Days to use to establish previous levels
PIP = 0.0001 # For EURUSD

COLS = ["DAY",
        "MONTH",
        "RET_BFVG",
        "RET_BFVG_D",
        "RET_bFVG",
        "RET_bFVG_P",
        
        "LIQ_LTH", # OK
        "LIQ_LTL", # OK
        
        "BFVG_VLTN_D",
        "bFVG_VLTN_P",
        
        "BTREND_D",
        "bTREND_D",
        
        "TRCSLD_P30M",
        
        "DAYOP_ABV", # OK
        "DAYOP_BLW", # OK
        "SESOP_ABV", # OK
        "SESOP_BLW"  # OK
        ] 

def updateMaxes(lst: list, m):
    
    for _ in range(len(lst)):
        if lst[-1] < m:
            lst.pop(-1)
        else:
            lst.append(m)
            break
    if len(lst) == 0:
        lst = [m]
    return(lst)

def updateMins(lst: list, m):
    for _ in range(len(lst)):
        if lst[-1] > m:
            lst.pop(-1)
        else:
            lst.append(m)
            break
    if len(lst) == 0:
        lst = [m]
    return(lst)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True)
    args = parser.parse_args()
    
    df_features = pd.DataFrame()
    
    m = Market(args.f)
    
    dailyHighs, dailyLows = [], [] # High/lows over the previous days that haven't been taken yet
    
    days = list(m.data["ts"].map(lambda x: str(x)[:10]).unique())
    
    if len(days) < HISTORY_DAYS:
        print("Not enough history days")
        exit(10)
    
    for day in range(HISTORY_DAYS):
        day_L = days[day].split("-")
        start, end = datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 0,0,0), datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 23, 59, 59)
        df = m.data[ (m.data["ts"] >= start) & (m.data["ts"] <= end) ]
        
        h,l = df["high"].max(), df["low"].min()
        dailyHighs = updateMaxes(dailyHighs, h)
        dailyLows = updateMins(dailyLows, l)
    
    days = days[HISTORY_DAYS:]
    
    print(dailyHighs, dailyLows, "\n")
    
    for day in range(len(days)):
        
        day_L = days[day].split("-")
        start, end = datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 0,0,0), datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 23, 59, 59)
        df = m.data[ (m.data["ts"] >= start) & (m.data["ts"] <= end) ]
        
        dayOpenPrice = df[df["ts"] == datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 0, 0, 0)]["open"]
        NYOpenPrice = df[df["ts"] == datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 7, 30, 0)]["open"]
        
        
        DAYOP_ABV = df[df['close'] > dayOpenPrice].shape[0]
        DAYOP_BLW = df[df['close'] < dayOpenPrice].shape[0]
        SESOP_ABV = df[df['close'] > NYOpenPrice].shape[0]
        SESOP_BLW = df[df['close'] < NYOpenPrice].shape[0]
        
        # Updating daily highs
        h,l = df["high"].max(), df["low"].min()
        highs_temp = updateMaxes(dailyHighs, h)
        LIQ_LTH = 1 if len(highs_temp) != len(dailyHighs)+1 else 0
        dailyHighs = highs_temp
        
        lows_temp = updateMins(dailyLows, l)
        LIQ_LTL = 1 if len(lows_temp) != len(dailyLows)+1 else 0
        dailyLows = lows_temp
        
        # FVG Violation
        BFVG_VLTN = 0
        BFVG_VLTN_D = 0
        bFVG_VLTN = 0
        bFVG_VLTN_P = 0
    
        BFVG_RET = 0
        BFVG_RET_D = 0
        bFVG_RET = 0
        bFVG_RET_P = 0
        
        mkt_day = Market(df)
        
        for fvg in mkt_day.fvg["bull"]:
            lim_low  = mkt_day.chart[fvg-1].high
            lim_high = mkt_day.chart[fvg+1].low
            
            pl = mkt_day.price_legs["bull"][fvg] # [[idx_start, idx_end], [low, high], [low_body, high_body]]
            
            equil = (pl[1][0] + pl[1][1])/2
            # equil_body = (pl[2][0] + pl[2][1])/2
            
            for c in range(fvg+1, len(mkt_day.chart)):
                if c.low < lim_high:
                    BFVG_RET += 1
                    
                if c.low < equil:
                    BFVG_RET_D += 1
                    
                if c.close < lim_low - 1*PIP: # 1 pip tolerance
                    BFVG_VLTN += 1
                    
                if c.close < (lim_low - 1*PIP) and lim_low < equil: # 1 pip tolerance
                    BFVG_VLTN_D += 1
        
        
        for fvg in mkt_day.fvg["bear"]:
            lim_high = mkt_day.chart[fvg-1].low
            lim_low  = mkt_day.chart[fvg+1].high
            
            pl = mkt_day.price_legs["bear"][fvg] # [[idx_start, idx_end], [low, high], [low_body, high_body]]
            
            equil = (pl[1][0] + pl[1][1])/2
            
            for c in range(fvg+1, len(mkt_day.chart)):
                if c.high > lim_low:
                    bFVG_RET += 1
                    
                if c.high > equil:
                    bFVG_RET_P += 1
                    
                if c.close > lim_high + 1*PIP: # 1 pip tolerance
                    bFVG_VLTN += 1
                    
                if c.close > (lim_high + 1*PIP) and lim_high > equil: # 1 pip tolerance
                    bFVG_VLTN_P += 1
        
        
        
        
        
    # m.plot(start=datetime.datetime(2023, 12, 5, 5, 0, 0), end=datetime.datetime(2023, 12, 6, 8, 0, 0), bFVG = False, trends=True)

