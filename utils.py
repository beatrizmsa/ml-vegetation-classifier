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
    # Create the subplots for each column
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    # Iterate through the columns of the dataset to create the boxplot
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
    _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

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


def barplot_correlation_visualization(data, title):
    plt.figure(figsize=(15, 10))
    sns.barplot(x=data.index, y=data.values, palette='viridis')
    plt.title(title)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def heatmap_visualization(data, title):
    plt.figure(figsize=(15, 15))
    sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f")
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
    _, axes = plt.subplots(
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
    # Iterate through the different models that we are going to validate
    for name, model in models.items():
        # Use training data to train the current model
        model.fit(X_train, y_train)

        # Predict the target variable with the trained model based on the test data
        y_pred = model.predict(X_test)

        # Create a classification report with the predict target variable
        # and the actual values
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extract chosen metrics from the classification report
        accuracy = report["accuracy"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]
        cm = confusion_matrix(y_test, y_pred)

        # Add results to the results data frame
        add_results(
            data_results, name, "Holdout" + suffix, accuracy, precision, recall, f1, "Nan", cm
        )


def cross_validation_evaluation(data_results, models, X, y, k_splits, suffix=""):
    # Create the cross validator with the given number of k_splits
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    
    # Iterate through models to apply k-fold cross validation to each one
    for name, model in models.items():
        # Create scaler to standardize the data
        scaler = StandardScaler()

        # Create pipeline to standardize and estimate the data
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])

        # Run k-fold cross validation for the current model
        # using the pipeline that was previously created
        scores = cross_validate(pipeline, X, y, cv=kf, scoring=metrics, n_jobs=-1)

        # Extract the chosen metrics from the scores given by the cross validation
        accuracy = scores["test_accuracy"].mean()
        precision = scores["test_precision_weighted"].mean()
        recall = scores["test_recall_weighted"].mean()
        f1 = scores["test_f1_weighted"].mean()
        std_dev = scores["test_accuracy"].std()

        # Predict the target variable to create a confusion matrix
        y_pred = cross_val_predict(pipeline, X, y, cv=kf, n_jobs=-1)

        # Create the confusion matrix with the results from the prediction
        cm = confusion_matrix(y, y_pred)

        # Add results to the data results data frame
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
    # Create the leave one out cross validator
    cv = LeaveOneOut()

    # Iterate through models to apply k-fold cross validation to each one
    for name, model in models.items():
        # Create scaler to standardize the data
        scaler = StandardScaler()

        # Create pipeline to standardize and estimate the data
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])

        # Run leave one out cross validation for the current model
        # using the pipeline that was previously created
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=metrics, n_jobs=-1)

        # Extract the chosen metrics from the scores given by the cross validation
        accuracy = scores["test_accuracy"].mean()
        precision = scores["test_precision_weighted"].mean()
        recall = scores["test_recall_weighted"].mean()
        f1 = scores["test_f1_weighted"].mean()
        std_dev = scores["test_accuracy"].std()

        # Predict the target variable to create a confusion matrix
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

        # Create the confusion matrix with the results from the prediction
        cm = confusion_matrix(y, y_pred)

        # Add results to the data results data frame
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


def bootstrap_evaluation(data_results, models, X, y, n_iterations=100, suffix=""):
    bootstrap_scores = {
        name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "cm": []}
        for name in models.keys()
    }

    for i in range(n_iterations):
        # Resample the data with bootstrapping strategy
        X_resampled, y_resampled = resample(
            X, y, n_samples=int(0.7 * len(X)), replace=True, random_state=42+i
        )

        # Get the id's of the data that was not chosen the resampling step
        test_idx = ~X.index.isin(X_resampled.index)

        # Create the test data using the id's from the previous step
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]


        scaler = StandardScaler()

        # Standardize the data using the StandardScaler
        X_resampled = scaler.fit_transform(X_resampled)
        X_test = scaler.transform(X_test)

        for name, model in models.items():
            # Fit the model to the resampled data
            model.fit(X_resampled, y_resampled)

            # Predict the target variable using the test data
            y_pred = model.predict(X_test)

            # Create classification report using the predicted and actual values
            report = classification_report(y_test, y_pred, output_dict=True)

            # Extract the chosen metrics from the classification report
            bootstrap_scores[name]["accuracy"].append(report["accuracy"])
            bootstrap_scores[name]["precision"].append(
                report["weighted avg"]["precision"]
            )
            bootstrap_scores[name]["recall"].append(report["weighted avg"]["recall"])
            bootstrap_scores[name]["f1"].append(report["weighted avg"]["f1-score"])
            # Create confusion matrix with the predicted and actual values
            # and add it to the bootstrap scores
            bootstrap_scores[name]["cm"].append(confusion_matrix(y_test, y_pred))

    # Iterate through the bootstrap scores
    for name, metrics in bootstrap_scores.items():
        # Calculate the mean of each metric for the current models
        accuracy = np.mean(metrics["accuracy"])
        precision = np.mean(metrics["precision"])
        recall = np.mean(metrics["recall"])
        f1 = np.mean(metrics["f1"])
        std_dev = np.std(metrics["accuracy"])

        # Sum the scores of the confusion matrices
        cm = np.sum(metrics["cm"], axis=0)

        # Add the results to the data results data frame
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


def apply_ridge(names, X_train, y_train, X_test, y_test):
    # Initializa K-fold cross validator
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

    # Initialize Ridge Regression model with built cross validator
    ridge = RidgeCV(
        alphas=np.arange(0.1, 10, 0.1), cv=cv, scoring='f1_weighted'
    )

    # Train model with the previously standardize training data
    ridge.fit(X_train, y_train)

    # Predict target variable with the previously standardized test data
    ridge_reg_y_pred = ridge.predict(X_test)

    # Print the metrics for the ridge model
    print("Ridge Tuning Parameter: ", (ridge.alpha_))
    print("Ridge Model Intercept: ", (ridge.intercept_))

    # Plot the coeficients with their respective columns
    plt.bar(names, ridge.coef_)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on Ridge")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

    print(
        "Ridge Regression Model RMSE is: ",
        np.sqrt(mean_squared_error(y_test, ridge_reg_y_pred)),
    )
    print(
        "Ridge Regression Model Training Score: ", ridge.score(X_train, y_train) * 100
    )
    print("Ridge Regression Model Testing Score: ", ridge.score(X_test, y_test) * 100)


def apply_lasso(names, X_train, y_train, X_test, y_test):
    # Initialize K-fold cross validator
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

    # Initialize Lasso model with built cross validator
    lasso = LassoCV(alphas=np.arange(0.1, 10, 0.1), cv=cv, tol=1)

    # Train model with the previously standardize training data
    lasso.fit(X_train, y_train)

    # Predict target variable with the previously standardized test data
    lasso_reg_y_pred = lasso.predict(X_test)

    # Print the metrics for the ridge model
    print("Lasso tuning parameter:", (lasso.alpha_))
    print("Lassso model intercept:", (lasso.intercept_))

    # Plot the coeficients with their respective columns
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
    # Initialize Elastic Net model with built cross validator
    enet = ElasticNetCV(alphas=np.arange(0.1, 10, 0.1), cv=10, l1_ratio=0.5)

    # Train model with the previously standardize training data
    enet.fit(X_train, y_train)

    # Predict target variable with the previously standardized test data
    enet_reg_y_pred = enet.predict(X_test)

    # Print the metrics for the elastic net model
    print("ElasticNet tuning parameter:", (enet.alpha_))
    print("ElasticNet model intercept:", (enet.intercept_))
    
    # Plot the coeficients with their respective columns
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
