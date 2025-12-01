import pandas as pd
from find_pairs import findPairs
from get_data import fetchStocks
from ols import strategy

pd.set_option('display.max_columns', None)

start = "2023-01-01"
end = "2025-01-01"
file = "stocks_sp500.csv"

loader = fetchStocks(file)
dfTickers = loader.getAllTickers()
df = loader.loadDataFromCSV()
pairs = findPairs(df, dfTickers).getPairs()

plt  = strategy(df, pairs[1]['stockA'], pairs[1]['stockB'])
plt.OLS()
plt.plotOLS()