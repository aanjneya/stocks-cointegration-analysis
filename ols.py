import statsmodels.api as sm
import matplotlib.pyplot as plt


class strategy:
    def __init__(self,df, stockA, stockB):
        self.df = df
        self.zscore = 0
        self.stockA = stockA
        self.stockB = stockB

    def OLS(self):
        X = self.df[self.stockA]
        Y = self.df[self.stockB]
        C= sm.add_constant(Y)

        model = sm.OLS(X,C).fit()

        hedge_ratio = model.params.iloc[1]
        spread =  X - (hedge_ratio * Y)
        roll_mean = spread.rolling(window=30).mean()
        roll_std = spread.rolling(window=30).std()
        self.z_score = (spread - roll_mean) / roll_std

    def plotOLS(self):
        self.z_score.plot()
        plt.axhline(2.0, color='red')
        plt.axhline(-2.0, color='green')
        plt.axhline(0, color='black')
        plt.show()