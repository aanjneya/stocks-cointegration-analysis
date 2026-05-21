import yfinance as yf
import pandas as pd

class FetchStocks:
    def __init__(self, file):
        self.start = ""
        self.end = ""
        self.file = file
        self.tickers = []

    def get_all_tickers(self):
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        self.tickers = tickers
        return df

    def get_tickers(self, ls):
        self.tickers = ls
        return self.tickers

    def fetch_data(self,start, end):
        self.start = start
        self.end = end
        df = yf.download(self.tickers,start = self.start,end = self.end)['Close']
        df = df.dropna(axis=1, how='all')
        df.to_csv(self.file)

    def load_data_from_csv(self):
        df = pd.read_csv(self.file,index_col=0, parse_dates=True)
        return df

if __name__ == "__main__":
    start = "2018-01-01"
    end = "2026-05-01"
    file = "stocks_sp500.csv"

    loader = FetchStocks(file)
    loader.get_all_tickers()
    loader.fetch_data(start, end)
