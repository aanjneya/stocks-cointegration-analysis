import pandas as pd
from statsmodels.tsa.stattools import coint
from get_data import fetchStocks

pd.set_option('display.max_columns', None)

start = "2023-01-01"
end = "2025-01-01"
file = "stocks_sp500.csv"

loader = fetchStocks(file)
dfTickers = loader.getAllTickers()
df = loader.loadDataFromCSV()

sectored_map = {}
for index, row in dfTickers.iterrows():
    ticker = row['Symbol']
    sector = row['GICS Sector']
    subsector = row['GICS Sub-Industry']

    if sector not in sectored_map:
        sectored_map[sector] = {}

    if subsector not in sectored_map[sector]:
        sectored_map[sector][subsector] = []

    sectored_map[sector][subsector].append(ticker)

pairs = []

for i in sectored_map.values():
    for substocks in i.values():
        if len(substocks) > 1:
            for j in range(len(substocks)):
                if substocks[j] not in df.columns:
                    continue
                stock1 = df[substocks[j]]
                for k in range(j+1, len(substocks)):
                    if substocks[k] not in df.columns:
                        continue
                    stock2 = df[substocks[k]]
                    merge = pd.concat([stock1,stock2], axis=1)
                    merge = merge.dropna()
                    if len(merge)>100:
                        pval = coint(merge.iloc[:, 0], merge.iloc[:, 1])[1]
                        if pval < 0.05:
                            pairs.append({'Stock_a':substocks[j], 'Stock_b':substocks[k], 'p-value':pval})

print(pairs)