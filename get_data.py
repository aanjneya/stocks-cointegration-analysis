import yfinance as yf
import pandas as pd
import requests
from io import StringIO

class fetchStocks:
    def __init__(self, file):
        self.start = ""
        self.end = ""
        self.file = file
        self.tickers = []

    def getAllTickers(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        r = requests.get(url, headers=headers)

        tables = pd.read_html(StringIO(r.text))

        df = None
        for table in tables:
            if 'Symbol' in table.columns and 'GICS Sector' in table.columns:
                df = table
                break

        if df is None:
            raise ValueError("Could not find the S&P 500 table in the Wikipedia response.")

        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        self.tickers = tickers
        return df

    def getTickers(self, ls):
        self.tickers = ls
        return self.tickers

    def fetchData(self,start, end):
        self.start = start
        self.end = end
        df = yf.download(self.tickers,start = self.start,end = self.end)['Close']
        df = df.dropna(axis=1, how='all')
        df.to_csv(self.file)

    def loadDataFromCSV(self):
        df = pd.read_csv(self.file,index_col=0, parse_dates=True)
        return df

if __name__ == "__main__":
    start = "2023-01-01"
    end = "2025-01-01"
    file = "stocks_sp500.csv"

    loader = fetchStocks(file)
    loader.getAllTickers(start, end)
    loader.fetchData()