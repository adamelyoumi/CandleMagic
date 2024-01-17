# CandleMagic

## Overview

We want to prove/disprove ICT concepts using Machine Learning 

## Part 1 - Dataset construction

Collect as many features as possible that relate to ICT concepts.

We want to populate our dataset with all days where a large price move occured in the 9:00 - 12:00 AM EST range, on a single pair. We may define a large move by a move of 40+ pips.

In addition, we need at least 10 trading days of history to establish reference high/low levels

We may work with 1M data only, as higher timeframes can be reconstructed easily, and since they provide the most fine-grained data. Features related to higher timeframes may be included.

<table>
<tr><th> Relevant Features </th></tr>
<tr><td>

|Name | Description | Type|
|--|--|--|
|RET_BFVG | Price retraced into a Bullish FVG| Boolean |
|RET_BFVG_D | Price retraced into a Bullish FVG in a Discount| Boolean |
|LIQ_LTH_X | Took long-term high (sell-side liquidity) (previous high/TBL)| Boolean |
|BFVG_VLTN | Price closed below a Bullish FVG| Boolean |
|BTREND_D | Daily bullish trend (past 3 days closed above open)| Boolean |
|TRC_P30M | A tight-range consolidation happened within the past 30 Minutes| Boolean |
|DAYOP_ABV | Price traded above the day opening (00:00 AM)| Boolean |
|SESOP_ABV | Price traded above the NY session opening (7:30 AM)| Boolean |
|Target variable|0 = No significant move, 1 = Bullish move, 2 = Bearish move| Boolean |

</td></tr> 
</table>

Features related to bearish price moves/buyside liquidity shall also be included.

A "_5M" suffix may be included to reference the same information present on a higher timeframe (here, the 5M timeframe as an example)

## Part 2 - Model Training

Define and train ML/DL models to observe the relevance of the collected metrics.

Once the correct model has been identified, different models may be trained on different pairs.

