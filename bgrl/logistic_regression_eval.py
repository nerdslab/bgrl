import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize


def fit_logistic_regression(X, y, data_random_seed=1, repeat=1):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    # set random state
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
                                                   # throughout training

    accuracies = []
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    print(np.mean(accuracies))
    return accuracies
