import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

sns.set_theme()


class exploratory_data_analysis:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # data description using pandas library    
    def describe(self):
        return self.df.describe()

    # count the cases of fraud or no fraud
    def rates(self):
        print("count of NO fraud cases:", (df["is_fraud"] == 0).sum())
        print("count of fraud cases:", (df["is_fraud"] == 1).sum())
        print("bad rate:", (df["is_fraud"] == 1).sum() / len(df) * 100, "%")

    # make correlation heatmap using sns
    def corr_heatmap(self):

        figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax1, ax2 = axes.flatten()

        ax1.title.set_text("Heatmap with Pearson Correlation")
        sns.heatmap(self.df.corr("pearson"), annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0])

        ax2.title.set_text("Heatmap with Spearman Correlation (For Bool  variables)")
        sns.heatmap(self.df.corr("spearman"), annot=True, fmt='.2f', cmap='Blues');

    # make boxplots of numerical variables
    def boxplots(self):

        cols = list(self.df.select_dtypes('float').columns) + ["city_pop"]
        print(cols)
        plt.figure(figsize=(15, 4))

        for i, col in enumerate(cols):
            ax = plt.subplot(1, len(cols), i + 1)
            sns.boxplot(data=self.df, x=col, ax=ax).set_title(col)

    # count int variables 
    def count_categorical(self, target_variable: Optional[str] = None):

        dtype = self.df.select_dtypes('int64').columns
        dtype = dtype[dtype != "city_pop"]

        if target_variable:
            data = self.df.copy()
            data = data[data[target_variable] == 1]
            dtype = dtype[dtype != target_variable]
            plt.figure(figsize=(15, 4))

            for i, col in enumerate(dtype):
                ax = plt.subplot(1, len(dtype), i + 1)
                sns.countplot(data=data, x=col, ax=ax).set_title(f"{col} when {target_variable}=1")
            plt.tight_layout()


        else:
            plt.figure(figsize=(15, 4))

            for i, col in enumerate(dtype):
                ax = plt.subplot(1, len(dtype), i + 1)
                sns.countplot(data=self.df, x=col, ax=ax).set_title(f"{col}")
            plt.tight_layout()

    # multivariable plot
    def multivariable_plot(self, dtypes: Optional[str] = None):

        data = self.df.copy()

        if dtypes:
            sns.pairplot(data=data[data.select_dtypes(dtypes).columns]);

        else:
            sns.pairplot(data=data);
