from classes import Candle, Market
import pandas as pd
import datetime
import plotly.graph_objects as ly
import matplotlib.pyplot as plt
import numpy as np
import bisect
import argparse

"""
    Name	Description
RET_BFVG	Price retraced into a Bullish FVG
RET_BFVG_D	Price retraced into a Bullish FVG in a Discount
LIQ_LTH_X	Took long-term high (sell-side liquidity) (previous high/TBL)
BFVG_VLTN	Price closed below a Bullish FVG
BTREND_D	Daily bullish trend (past 3 days closed above open)
TRC_P30M	A tight-range consolidation happened within the past 30 Minutes
DAYOP_ABV	Price traded above the day opening (00:00 AM)
SESOP_ABV	Price traded above the NY session opening (7:30 AM)

Target variable : 0 = No significant move, 1 = Bullish move, 2 = Bearish move
"""

BIG_MOVE = 40 # pips
HISTORY_DAYS = 10 # Days to use to establish previous levels

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    args = parser.parse_args(args=[])
    
    m = Market(args.f)
    
    dailyHighs, dailyLows = [], []
    
    days = list(m.data["ts"].map(lambda x: str(x)[:10]).unique())
    
    if len(days) < HISTORY_DAYS:
        print("Not enough history days")
        exit(10)
    
    for day in range(HISTORY_DAYS):
        day_L = days[day].split("-")
        start, end = datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 0,0,0), datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 23, 59, 59)
        df = m.data[ (m.data["ts"] >= start) & (m.data["ts"] <= end) ]
        
        h,l = df["high"].max(), df["low"].min()
        idH, idL = bisect.bisect(dailyHighs, h), bisect.bisect(dailyLows, l)
        dailyHighs.insert(idH, h)
        dailyHighs = dailyHighs[:idH+1] # Remove the daily highs that have been taken out
        
        dailyLows.insert(idL, l)
        dailyLows = dailyLows[:idL+1]
    
    days = days[HISTORY_DAYS:]
    
    print(dailyHighs, "\n", dailyLows)
    
    for day in range(len(days)):
        dayOpenPrice = m.data[m.data["ts"] == datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 0, 0, 0)]["open"]
        NYOpenPrice = m.data[m.data["ts"] == datetime.datetime(int(day_L[0]), int(day_L[1]), int(day_L[2]), 7, 30, 0)]["open"]
    
    
    
    
    
    # m.plot(start=datetime.datetime(2023, 12, 5, 5, 0, 0), end=datetime.datetime(2023, 12, 6, 8, 0, 0), bFVG = False, trends=True)
    