import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.base import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, cross_val_predict, cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn import tree
from pygam import LogisticGAM, s, f, l


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
    fig, axes = plt.subplots(int(np.ceil(len(data) / 3)), 3, figsize=(15, 15))

    axes = axes.flatten()
    for i, row in data.iterrows():
        sns.heatmap(
            row["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[i]
        )
        axes[i].set_title(f'{row["Model"]} using {row["Method"]}')
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    for j in range(len(data), len(axes)):
        axes[j].axis("off")

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


def holdout_evaluation(data_results, models, X, y, suffix=""):
    holdout_scores = {
        name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "cm": []}
        for name in models.keys()
    }
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 + i)
        scaller = StandardScaler()
        X_train = scaller.fit_transform(X_train)
        X_test = scaller.transform(X_test)
        for name, model in models.items():
            # Use training data to train the current model
            model.fit(X_train, y_train)

            # Predict the target variable with the trained model based on the test data
            y_pred = model.predict(X_test)

            # Create a classification report with the predict target variable
            # and the actual values
            report = classification_report(y_test, y_pred, output_dict=True)

            # Extract chosen metrics from the classification report
            holdout_scores[name]["accuracy"].append(report["accuracy"])
            holdout_scores[name]["precision"].append(report["weighted avg"]["precision"])
            holdout_scores[name]["recall"].append(report["weighted avg"]["recall"])
            holdout_scores[name]["f1"].append(report["weighted avg"]["f1-score"])
            holdout_scores[name]["cm"].append(confusion_matrix(y_test, y_pred))

    for name, metrics in holdout_scores.items():
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
            "Houldout" + suffix,
            accuracy,
            precision,
            recall,
            f1,
            std_dev,
            cm,
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
            X, y, n_samples=int(len(X)), replace=True, random_state=42+i
        )

        # Get the id's of the data that was not chosen in the resampling step
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


def best_feature_grid_search_visualization(X_train, y_train, parameters, model, result):

    # initialize the GridSeachCV method
    grid = GridSearchCV(model, parameters, cv=KFold(n_splits=5, shuffle=True, random_state=42))

    # fit with the training data
    grid.fit(X_train, y_train)

    # substitute the score with a better one
    if result['C']:
        if grid.best_score_ >= result['score']:
            result['C'] = grid.best_params_['C']
            result['score'] = grid.best_score_
            # store "l1_ratio" parameter for the Elastic Net method
            if 'l1_ratio' in parameters.keys():
                result['l1_ratio'] = grid.best_params_['l1_ratio']

    # initialize the C parameters for the first time
    else:
        result['C'] = grid.best_params_['C']
        result['score'] = grid.best_score_
        # store "l1_ratio" parameter for the Elastic Net method
        if 'l1_ratio' in parameters.keys():
            result['l1_ratio'] = grid.best_params_['l1_ratio']

    # store all the scores and parameters
    scores = grid.cv_results_['mean_test_score']
    params = grid.cv_results_['params']

    # create dataframe with all the results and plot the graph
    results_df = pd.DataFrame(params)
    results_df['mean_test_score'] = scores
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['C'], results_df['mean_test_score'], marker='o')
    plt.title('Score vs. C')
    plt.xlabel('C')
    plt.ylabel('Mean Test Score')
    plt.grid()
    plt.show()

    print("Best parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)

    return result


def best_feature_grid_search(columns_name, X_train, y_train, parameters, model):
    # initialize the GridSeachCV method
    grid = GridSearchCV(model, parameters, cv=KFold(n_splits=5, shuffle=True, random_state=42))

    # fit with the training data
    grid.fit(X_train, y_train)
    coef = grid.best_estimator_.coef_[0]

    print("Best parameters:", grid.best_params_)

    # print the features and the respective coefficients
    print("Feature Coefficients:")
    print(pd.DataFrame({'Feature': columns_name, 'Coefficient': coef}))
    return grid.best_params_


def gam_gridsearch(X_data, y_data, configs):
    # Prepare for cross-validation with 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_score = 0
    best_scores = []
    best_model = None

    # Iterate over each configuration
    for config in configs:
        cv_scores = []

        # Manually iterate over each fold in KFold
        for train_index, test_index in kf.split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            gam = LogisticGAM(config)
            # Fit the model on the training data
            gam.fit(X_train, y_train)

            # Predict on the test data and evaluate F1 score
            y_pred = gam.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            cv_scores.append(f1)

        # Calculate mean F1 score for the current configuration
        mean_score = np.mean(cv_scores)

        # Update the best model if this configuration has a higher mean F1 score
        if mean_score > best_score:
            best_score = mean_score
            best_model = gam
            best_scores = cv_scores

    # Output the best model configuration
    print("Best Model Configuration:", best_model.terms)
    print(f"Cross-validation F1 scores: {best_scores}")
    print(f"Best Mean F1 Score: {best_score:.4f}")
    print(f"Standard deviation of F1 score: {np.std(best_scores):.4f}")

### Second Part

def get_feature_importances_text(column_names, importances):
    features = list(zip(column_names, importances))

    features_sorted = sorted(features, key=lambda x: x[1], reverse=True)
    print("Feature Importances:")
    for feature, importance in features_sorted:
        print(f"- {feature}: {importance:.3f}")


def best_random_forest(X_train, y_train, param_rfm):
    rfm = RandomForestClassifier()

    rfm_gs = GridSearchCV(rfm, param_rfm, cv=5, scoring='accuracy')
    rfm_gs.fit(X_train, y_train)
    print("Best parameters", rfm_gs.best_params_)

def decision_tree(X_data, y_data):
    dt = DecisionTreeClassifier(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    dt.fit(X_train, y_train)

    print("Max depth:", dt.tree_.max_depth)
    print("Number of nodes:", dt.tree_.node_count)

    print('Training set score: {:.4f}'.format(dt.score(X_train, y_train)))

    print('Test set score: {:.4f}'.format(dt.score(X_test, y_test)))


def decision_tree_grid_search(X_data, y_data, params, generate_tree=False):
    dt = DecisionTreeClassifier()

    gs = GridSearchCV(
        dt,
        params,
        cv=5,
        scoring='accuracy',
    )

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    gs.fit(X_train, y_train)

    print("Best Hyperparameters:", gs.best_params_)
    best_model = gs.best_estimator_

    print('Training set score: {:.4f}'.format(best_model.score(X_train, y_train)))

    best_model_score = best_model.score(X_test, y_test)

    print('Test set score: {:.4f}'.format(best_model_score))

    if generate_tree:
        dot_data = tree.export_graphviz(
            best_model,
            out_file=None,
            feature_names=X_train.columns,
            class_names=np.array(sorted(y_train.unique())).astype('str').tolist(),
            filled=True,
        )
        
        graph = graphviz.Source(dot_data)

        return graph


def q_learning_feature_selection(
    X, y,
    learning_rate=0.8,
    discount_factor=0.95,
    episodes=100,
    epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay=0.995
):
    n_features = X.shape[1]

    # Create a Q-table with rows as states (possible feature combinations)
    # and columns as actions (features to select or stop)
    Q_table = np.zeros((2**n_features, n_features + 1))

    def state_to_index(state):
        return int("".join(map(str, state.astype(int))), 2)

    rewards = []

    for _ in range(episodes):
        # For each episode, start with an empty feature set
        state = np.zeros(n_features, dtype=int)
        total_reward = 0

        for _ in range(n_features):
            state_idx = state_to_index(state)

            # Choose actions based on an epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(np.where(state == 0)[0].tolist() + [n_features])  # Select a feature or stop
            else:
                action = np.argmax(Q_table[state_idx, :])

            if action == n_features:  # Stop action
                break

            # Update the state
            state[action] = 1
            selected_features = np.where(state == 1)[0]

            # Train and evaluate model
            X_train, X_test, y_train, y_test = train_test_split(X[:, selected_features], y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            accuracy = accuracy_score(y_test, model.predict(X_test))

            # Reward after evaluating the classification model with the
            # selected features and penalize based on the number of features
            reward = accuracy - 0.01 * len(selected_features)

            next_state_idx = state_to_index(state)
            # Update the Q-table using the Q-learning equation
            Q_table[state_idx, action] += learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state_idx, :]) - Q_table[state_idx, action]
            )

            total_reward += reward

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards.append(total_reward)

    return Q_table, rewards
