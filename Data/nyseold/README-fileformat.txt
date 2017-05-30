############################################
# Description of the {nyse,nysemerged}.zip #
############################################

----------------------------------------
File format: csv (Comma Separated Value)
----------------------------------------
The file name is in the following format: "stock's name".csv.
First column contains the date in YYMMDD format and the second column
is daily price (multiplier) of the given stocks. The daily price
is the fraction of the stock exchange closing prices on two consecutive
market days. The field separator is ",".

----------------------------
nyse.zip (from Yoram Singer)
----------------------------
This data set includes daily prices of 36 assets.
from: 1962.07.03
until: 1984.12.31
number of the trading days: 5651

stocks' and tickers' name in: "NyseTicker.csv".


-----------------------------------------------------------------------
nysemerged.zip (from Yahoo! Finance, data cleaning and data preparation
                            by Ga'bor Gelencse'r)
-----------------------------------------------------------------------
This data set includes daily prices of 23 assets.
from: 1962.07.03
until: 2006.11.24
number of the trading days: 11178

From 1962.07.03 to 1984.12.31 the data is identical to the nyse.zip data
set. After that period, the daily price multiplier (X(i)) was calculated
from the nominal values of closing prices, dividends and splits time series
by the following formula for all trading day 'i':

X(i) = Closing_Price(i) / (Closing_Price(i-1)-DIV(i)) 
       			* Split_multiplier(i) / (Stock_divident_multiplier(i)+1)

E.g. a 5:2 split on trading day 'i' is a 2.5 multiplier in the formula,
no split means 1. For more examples see
http://www.investopedia.com/ask/answers/06/adjustedclosingprice.asp

For missing trading days, X(i)=1.

The remaining 2 time series are the cash asset (which is constant X(i)=1
for all trading day 'i') and the "bond" asset, which is calculated from
the O/N Fed funds rates available at
http://www.federalreserve.gov/releases/h15/data/Daily/H15_FF_O.txt
The main difference between the bond and the stocks that X(i) for bond is known
on trading day 'i-1', while those for stocks aren't.


stocks' and tickers' name in: "NyseTickerMerged.csv".

