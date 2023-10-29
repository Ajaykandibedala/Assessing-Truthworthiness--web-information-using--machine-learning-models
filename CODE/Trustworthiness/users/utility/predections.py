from sklearn.model_selection import train_test_split
from django.conf import settings
import pandas as pd
from sklearn import tree

path = settings.MEDIA_ROOT + "//" + "heart_obecity.csv"
df = pd.read_csv(path)
X = df.iloc[:, :-1].values  # indipendent variable
y = df.iloc[:, -1].values  # Dependent variable


def fitting(X, y, criterion, splitter, mdepth, clweight, minleaf):
    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Fit the model
    model = tree.DecisionTreeClassifier(criterion=criterion,
                                        splitter=splitter,
                                        max_depth=mdepth,
                                        class_weight=clweight,
                                        min_samples_leaf=minleaf,
                                        random_state=0,
                                        )
    clf = model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, clf


def test_user_data(data):
    X_train, X_test, y_train, y_test, clf = fitting(X, y, 'gini', 'best', mdepth=3, clweight=None, minleaf=1000)
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict([data])
    return y_pred
