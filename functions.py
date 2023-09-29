import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import classification_report
from sklearn.metrics.cluster import contingency_matrix

from typing import Optional, List
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
        print(f"{target_variable} rate:", (self.df[target_variable] == 1).sum() / len(self.df) * 100, "%")

    # make correlation heatmap using sns
    def corr_heatmap(self):

        figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax1, ax2 = axes.flatten()

        ax1.title.set_text("Heatmap with Pearson Correlation")
        sns.heatmap(self.df.corr("pearson"), annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0])

        ax2.title.set_text("Heatmap with Spearman Correlation (For Bool variables)")
        sns.heatmap(self.df.corr("spearman"), annot=True, fmt='.2f', cmap='Blues');

    # make boxplots of numerical variables
    def boxplots(self, target_variable: Optional[str]=None):

        data=self.df.copy()

        if target_variable:
            data=data[data["is_fraud"]==1]


        cols = list(self.df.select_dtypes('float').columns)+ ["city_pop"]
        plt.figure(figsize=(15, 4))

        for i, col in enumerate(cols):
            ax = plt.subplot(1, len(cols), i + 1)
            sns.boxplot(data=data, x=col, ax=ax).set_title(col)

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


class diagnosis:

    def __init__(self, data, model, y, probas):
        self.data = data
        self.model = model
        self.y = y
        self.probs = probas

    def residual_analysis(self):
        residuals = self.y - self.probs
        plt.figure(figsize=(15, 6))
        plt.scatter(self.model.fittedvalues, residuals)
        plt.axhline(np.std(residuals) * -2, label="$\pm 2 \sigma^2$", color='red', linestyle='--')
        plt.axhline(0, linestyle='--', label="0", color="green")
        plt.axhline(np.std(residuals) * 2, color='red', linestyle='--')
        plt.xlabel("y_hat")
        plt.ylabel("errors")
        plt.title("Residual Analysis")
        plt.legend()
        plt.show();

    def cook(self):
        model_influence = self.model.get_influence()
        distance = model_influence.cooks_distance[0]
        p_value = model_influence.cooks_distance[1]

        plt.figure(figsize=(15, 6))
        plt.bar(range(1, len(self.data) + 1), distance, color="red")
        plt.xticks(rotation=90)
        plt.title("Leverage Analysis")
        plt.xlabel("Observation")
        plt.ylabel('Cooks Distance')
        plt.show()

    def distribution_prob(self, target_variable):
        data = self.data.copy()
        data['predicted_probs'] = self.probs
        plt.figure(figsize=(15, 6))
        sns.kdeplot(data=data, x='predicted_probs', hue=target_variable, fill=True)

        # Agrega etiquetas y título al gráfico
        plt.xlabel('Estimated Prob')
        plt.ylabel('Density')
        plt.title('Probability Distribution by category')

        # Muestra el gráfico
        plt.show();

    def residual_deviance(self):
        # Obtén los residuos deviance del modelo
        residuals = self.model.resid_dev.copy()
        plt.figure(figsize=(15, 6))
        # Crea un gráfico de barras de residuos deviance
        plt.bar(range(len(residuals)), residuals, color='red')

        # Agrega una línea horizontal en y=0 para referencia
        plt.axhline(y=0, color='r', linestyle='--')

        # Agrega etiquetas y título al gráfico
        plt.xlabel('Observation')
        plt.ylabel('Residual Deviance')
        plt.title('Residual Deviance Graphic')

        # Muestra el gráfico
        plt.show()
