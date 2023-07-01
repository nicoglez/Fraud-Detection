import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List
from sklearn.metrics import classification_report
from sklearn.metrics.cluster import contingency_matrix

sns.set_theme()


class exploratory_data_analysis:

    def __init__(self, df):
        self.df = df

    # data description using pandas library
    def describe(self):
        return self.df.describe()

    # count the cases of fraud or no fraud
    def rates(self, target_variable: Optional[str]):
        print(f"count of 0 cases:", (self.df[target_variable] == 0).sum())
        print("count 1 cases:", (self.df[target_variable] == 1).sum())
        print("bankruptcie rate:", (self.df[target_variable] == 1).sum() / len(self.df) * 100, "%")

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

        cols = list(self.df.select_dtypes('float').columns)
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


class logistic_class:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # Function to try a given model
    def log_model(self, target_variable: str, add_intercept: Optional[bool] = True,
                  ignore_variables: Optional[List] = None, return_model: Optional[bool] = False):

        data = self.df.copy()

        # Add intercept if neccesary
        if add_intercept:
            data["intercept"] = 1

        # Separate target and predictors
        target = data[target_variable]
        predictors = data.drop(target_variable, axis=1)

        if ignore_variables:
            predictors.drop(ignore_variables, axis=1, inplace=True)

        # make model
        logit_model = sm.Logit(target, predictors)
        logit_result = logit_model.fit()
        print(logit_result.summary())
        print()
        print(f"G={2 * np.log(logit_result.llnull / logit_result.llf)}")

        # predictions
        y_hat = logit_result.predict(predictors)
        y_hat_f = []
        for i in y_hat:
            if i > 0.5:
                y_hat_f.append(1)
            else:
                y_hat_f.append(0)

        if return_model:
            self.y_true = target
            self.y_hat = y_hat_f
            return logit_result, y_hat_f, y_hat, target

    # show contingency table
    def contingency_table(self):
        print("Contingency Matrix")
        print(contingency_matrix(self.y_true, self.y_hat))

    # show report
    def report(self):
        print(classification_report(self.y_true, self.y_hat))


