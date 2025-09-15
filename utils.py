from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def logreg_y_test_pred(df_train, df_test, features):
    """
    Train a logistic regression model on standardized features and predict test set labels.
    """
    X_train = df_train[features]
    y_train = df_train['Quality_Label']
    X_test = df_test[features]
    y_test = df_test['Quality_Label']
    # Standarizate train (fit + transform)
    # Standarizate test (transform)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Regression train (fit)
    # Regression test (predict)
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    # Predict on test set
    y_test_pred = logreg.predict(X_test_scaled)
    return y_test_pred

def logreg_accuracy(df, train_test, features, target):
    """
    Calculate logistic regression accuracy with a bootstraping aproximation
    """
    acc = []
    for train, test in train_test:
        # build data train/test
        df_train = df.iloc[train]
        df_test = df.iloc[test]
        y_test = df_test[target]
        y_test_pred = logreg_y_test_pred(df_train, df_test, features)
        acc.append(accuracy_score(y_test, y_test_pred))
    return acc