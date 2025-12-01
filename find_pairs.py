from statsmodels.tsa.stattools import coint
import pandas as pd

class findPairs:

    def __init__(self, df, dfTickers):
        self.df = df
        self.dfTickers = dfTickers
        self.sectored_map = {}
        self.pairs = []

    def getPairs(self):
        for index, row in self.dfTickers.iterrows():
            ticker = row['Symbol']
            sector = row['GICS Sector']
            subsector = row['GICS Sub-Industry']

            if sector not in self.sectored_map:
                self.sectored_map[sector] = {}

            if subsector not in self.sectored_map[sector]:
                self.sectored_map[sector][subsector] = []

            self.sectored_map[sector][subsector].append(ticker)

        for i in self.sectored_map.values():
            for substocks in i.values():
                if len(substocks) > 1:
                    for j in range(len(substocks)):
                        if substocks[j] not in self.df.columns:
                            continue
                        stock1 = self.df[substocks[j]]
                        for k in range(j+1, len(substocks)):
                            if substocks[k] not in self.df.columns:
                                continue
                            stock2 = self.df[substocks[k]]
                            merge = pd.concat([stock1,stock2], axis=1)
                            merge = merge.dropna()
                            if len(merge)>100:
                                pval = coint(merge.iloc[:, 0], merge.iloc[:, 1])[1]
                                if pval < 0.05:
                                    self.pairs.append({'stockA':substocks[j], 'stockB':substocks[k], 'pValue':pval})
        return self.pairs
