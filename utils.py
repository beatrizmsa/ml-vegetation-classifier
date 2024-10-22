# Recebe um dataset e faz alteracoes e avalia cada
# dataset com diversos modelos printa os resultados
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


colors = [
    "#FFB3BA",
    "#FFDFBA",
    "#FFFFBA",
    "#BAFFC9",
    "#BAE1FF",
    "#FFB3E6",
    "#B3FFDA",
    "#CAB2FF",
    "#FFB3FF",
    "#FFC1E3",
    "#CCE5FF",
    "#B2F0E6",
    "#FFD1B2",
    "#FFFF99",
    "#D1C4E9",
    "#FFE0B2",
    "#F8BBD0",
    "#DCEDC8",
]

metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]


def boxplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(data=data, y=col, ax=axes[i], color=colors[i])
        axes[i].set_title(col)
        axes[i].set_ylabel("")

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def boxplot_by_type_visualization(data, columns, title):
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(x="Vegetation_Type", y=col, data=data, ax=axes[i], color=colors[i])
        axes[i].set_xlabel("Vegetation Type")
        axes[i].set_ylabel(col)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def barplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        label = data.groupby(col).size()
        sns.barplot(x=label.index, y=label.values, ax=axes[i], color=colors[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def crosstab_by_type_visualization(data, columns, title):
    columns = [col for col in columns if col != "Vegetation_Type"]
    n_cols = min(1, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        crosstab_result = pd.crosstab(data[col], data["Vegetation_Type"])
        crosstab_result.plot(kind="bar", ax=axes[i])
        axes[i].set_xlabel("Vegetation Type")
        axes[i].set_ylabel(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def confusion_matrix_visualization(data, title):
    fig, axes = plt.subplots(
        data["Method"].nunique(), data["Model"].nunique(), figsize=(15, 15)
    )

    axes = axes.flatten()
    for i, row in data.iterrows():
        sns.heatmap(
            row["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[i]
        )
        axes[i].set_title(f'{row["Model"]} using {row["Method"]}')
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def add_results(
    data_results, model_name, method_name, accuracy, precision, recall, f1, std, cm
):
    new_row = {
        "Model": model_name,
        "Method": method_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Std Dev": std,
        "Confusion Matrix": cm,
    }
    data_results.loc[len(data_results)] = new_row


def holdout_evaluation(data_results, models, X_train, X_test, y_train, y_test, suffix=""):
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        accuracy = report["accuracy"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]
        cm = confusion_matrix(y_test, y_pred)
        add_results(
            data_results, name, "Holdout" + suffix, accuracy, precision, recall, f1, "Nan", cm
        )


def cross_validation_evaluation(data_results, models, X, y, k_splits, suffix=""):
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    for name, model in models.items():
        scaler = StandardScaler()
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
        scores = cross_validate(pipeline, X, y, cv=kf, scoring=metrics, n_jobs=-1)

        accuracy = scores["test_accuracy"].mean()
        precision = scores["test_precision_weighted"].mean()
        recall = scores["test_recall_weighted"].mean()
        f1 = scores["test_f1_weighted"].mean()
        std_dev = scores["test_accuracy"].std()

        y_pred = cross_val_predict(pipeline, X, y, cv=kf, n_jobs=-1)

        cm = confusion_matrix(y, y_pred)

        add_results(
            data_results,
            name,
            f"Cross-Validation with {k_splits}" + suffix,
            accuracy,
            precision,
            recall,
            f1,
            std_dev,
            cm,
        )


def loocv_evaluation(data_results, models, X, y, suffix=""):
    cv = LeaveOneOut()

    for name, model in models.items():
        scaler = StandardScaler()
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])

        scores = cross_validate(pipeline, X, y, cv=cv, scoring=metrics, n_jobs=-1)

        accuracy = scores["test_accuracy"].mean()
        precision = scores["test_precision_weighted"].mean()
        recall = scores["test_recall_weighted"].mean()
        f1 = scores["test_f1_weighted"].mean()
        std_dev = scores["test_accuracy"].std()

        y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

        cm = confusion_matrix(y, y_pred)

        add_results(
            data_results,
            name,
            "LeaveOneOut" + suffix,
            accuracy,
            precision,
            recall,
            f1,
            std_dev,
            cm,
        )


def bootstrap_evaluation(data_results, models, X, y , n_iterations=100, suffix=""):
    bootstrap_scores = {
        name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "cm": []}
        for name in models.keys()
    }

    for i in range(n_iterations):

        X_resampled, y_resampled = resample(
            X, y, n_samples=int(0.7 * len(X)), replace=True, random_state=42
        )
        test_idx = ~X.index.isin(X_resampled.index)

        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        scaler = StandardScaler()
        X_resampled = scaler.fit_transform(X_resampled)
        X_test = scaler.transform(X_test)


        for name, model in models.items():
            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            bootstrap_scores[name]["accuracy"].append(report["accuracy"])
            bootstrap_scores[name]["precision"].append(
                report["weighted avg"]["precision"]
            )
            bootstrap_scores[name]["recall"].append(report["weighted avg"]["recall"])
            bootstrap_scores[name]["f1"].append(report["weighted avg"]["f1-score"])
            bootstrap_scores[name]["cm"].append(confusion_matrix(y_test, y_pred))

    for name, metrics in bootstrap_scores.items():
        accuracy = np.mean(metrics["accuracy"])
        precision = np.mean(metrics["precision"])
        recall = np.mean(metrics["recall"])
        f1 = np.mean(metrics["f1"])
        std_dev = np.std(metrics["accuracy"])
        cm = np.sum(metrics["cm"], axis=0)

        add_results(
            data_results,
            name,
            "Bootstrap" + suffix,
            accuracy,
            precision,
            recall,
            f1,
            std_dev,
            cm,
        )


def apply_ridge_regression(names, X_train, y_train, X_test, y_test):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    ridge = RidgeCV(
        alphas=np.arange(0.1, 10, 0.1), cv=cv, scoring='f1_weighted'
    )
    print(ridge)

    ridge.fit(X_train, y_train)
    ridge_reg_y_pred = ridge.predict(X_test)
    print("Ridge Tuning Parameter: ", (ridge.alpha_))
    print("Ridge Model Intercept: ", (ridge.intercept_))
    plt.bar(names, ridge.coef_)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on Ridge")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

    important_features = [name for coef, name in zip(ridge.coef_, names) if abs(coef) > 0.1]
    print("Columns with coefficients above 0.1:", important_features)
    important_features = [name for coef, name in zip(ridge.coef_, names) if abs(coef) > 0.4]
    print("Columns with coefficients below 0.4:", important_features)

    print(
        "Ridge Regression Model RMSE is: ",
        np.sqrt(mean_squared_error(y_test, ridge_reg_y_pred)),
    )
    print(
        "Ridge Regression Model Training Score: ", ridge.score(X_train, y_train) * 100
    )
    print("Ridge Regression Model Testing Score: ", ridge.score(X_test, y_test) * 100)


def apply_lasso(names, X_train, y_train, X_test, y_test):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    lasso = LassoCV(alphas=np.arange(0.1, 10, 0.1), cv=cv, tol=1)

    lasso.fit(X_train, y_train)
    lasso_reg_y_pred = lasso.predict(X_test)
    print("Lasso tuning parameter:", (lasso.alpha_))
    print("Lassso model intercept:", (lasso.intercept_))
    plt.bar(names, lasso.coef_)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on Lasso")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

    print(
        "Lasso Regression Model RMSE is: ",
        np.sqrt(mean_squared_error(y_test, lasso_reg_y_pred)),
    )
    print(
        "Lasso Regression Model Training Score: ", lasso.score(X_train, y_train) * 100
    )
    print("Lasso Regression Model Testing Score: ", lasso.score(X_test, y_test) * 100)


def apply_elastic_net(names, X_train, y_train, X_test, y_test):
    enet = ElasticNetCV(alphas=np.arange(0.1, 10, 0.1), cv=10, l1_ratio=0.5)
    enet.fit(X_train, y_train)
    enet_reg_y_pred = enet.predict(X_test)

    print("ElasticNet tuning parameter:", (enet.alpha_))
    print("ElasticNet model intercept:", (enet.intercept_))
    plt.bar(names, enet.coef_)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on Elastic Net")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

    print(
        "ElasticNet Regression Model RMSE is: ",
        np.sqrt(mean_squared_error(y_test, enet_reg_y_pred)),
    )
    print(
        "ElasticNet Regression Model Training Score: ",
        enet.score(X_train, y_train) * 100,
    )
    print(
        "ElasticNet Regression Model Testing Score: ", enet.score(X_test, y_test) * 100
    )
