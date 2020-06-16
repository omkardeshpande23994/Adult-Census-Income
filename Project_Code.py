import timeit
import warnings
from heapq import nlargest
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn.compose._column_transformer import (
    ColumnTransformer,
    make_column_transformer,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.model_selection._validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, plot_importance


# For SKLEARN warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore", category=FutureWarning)


########### File Handling ###########

# Reading data into pandas dataset
data_all = pd.read_csv("data.csv", low_memory=False)

print("Dataset provided : \n", data_all.head())
print("\nShape of Dataset : ", data_all.shape)

cols_list = list(data_all.columns.values)


###########  Preprocessing  ###########

# creating target labels
data_all["income"] = data_all["income"].map({"<=50K": 0, ">50K": 1})

# Dropping unncessary columns
data_all.drop(["education"], axis=1, inplace=True)

# Exploring the Data

# Graph for Income count
sns.countplot(data_all["income"], label="Count")
plt.show()

# Explore Education vs Income
g = sns.factorplot(
    x="education.num", y="income", data=data_all, kind="bar", size=6, palette="muted"
)
g.despine(left=True)
g = g.set_ylabels(">50K probability")

# Explore Age vs Income
g = sns.FacetGrid(data_all, col="income")
g = g.map(sns.distplot, "age")
plt.show()

# Explore Sex vs Income
g = sns.barplot(x="sex", y="income", data=data_all)
g = g.set_ylabel("Income >50K Probability")
plt.show()

# Explore Marital Status vs Income
g = sns.factorplot(
    x="marital.status", y="income", data=data_all, kind="bar", size=6, palette="muted"
)
g.despine(left=True)
g = g.set_ylabels("Income >50K Probability")
plt.show()

# Race distribution
data_all.groupby(["race"]).size().plot(kind="bar", fontsize=14)
plt.show()

# Heatmap
hmap = data_all.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=0.8, annot=True, cmap="BrBG", square=True)
plt.show()


### 1.1 Outlier detection ###
# No outliers in categorical Data

### 1.2 Missing values ###
# replace blanks and ? with none
data_all.replace(["?", " ", "NULL"], np.nan, inplace=True)

# Check the percentage of Data missing/Null values
print("\nPercent of missing values : \n", data_all.isnull().mean())

# Option 1 : Dropping Nan values
# X.dropna(axis=1, inplace=True)

# Option 2 : imputing the Nan data with most frequent and mean for numeric
categoricalFeatures = [
    "workclass",
    "occupation",
    "native.country",
    "relationship",
    "race",
    "sex",
    "marital.status",
]

numericalFeatures = [
    "age",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "fnlwgt",
]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat1", SimpleImputer(strategy="mean"), numericalFeatures),
        ("num", SimpleImputer(strategy="most_frequent"), categoricalFeatures),
    ],
    remainder="passthrough",
)

imputed_tr = preprocessor.fit_transform(data_all)
imputed_data = pd.DataFrame(data=imputed_tr)
# appending column names
imputed_data.columns = numericalFeatures + categoricalFeatures + ["Class_label"]

print("\nData after imputing : \n")
print(imputed_data.head())

### 1.3 Handling categorical data ###


# mapping feature values in marital class
print("\nValues in Marital Status :\n", imputed_data["marital.status"].unique())

imputed_data["marital.status"] = imputed_data["marital.status"].map(
    {
        "Married-civ-spouse": "Married",
        "Divorced": "Single",
        "Never-married": "Single",
        "Separated": "Single",
        "Widowed": "Single",
        "Married-spouse-absent": "Married",
        "Married-AF-spouse": "Married",
    }
)

# Converting status to labels
imputed_data["marital.status"] = imputed_data["marital.status"].map(
    {"Single": 0, "Married": 1}
)

# Converting sex to labels
imputed_data["sex"] = imputed_data["sex"].map({"Male": 0, "Female": 1})

# creating features and labels
X = imputed_data.iloc[:, :-1]
y = imputed_data.iloc[:, -1]


# # checking numeric and categorical data
print(X.head())

# numeric = list(range(0, 6))
# categorical = list(range(6, 11))

# print(numeric, categorical)

preprocessor = ColumnTransformer(
    [
        ("onehot", OneHotEncoder(), categoricalFeatures),
        # ("ordinal", OrdinalEncoder(), categoricalFeatures),
        ("scaler", StandardScaler(), numericalFeatures),
    ],
    remainder="passthrough",
)


encoded_tr = preprocessor.fit_transform(X)
encoded_tr = encoded_tr.toarray()
encoded_data = pd.DataFrame(data=encoded_tr)
print("\nData after encoding : \n")
print(encoded_data.head())


X = encoded_data
y = y.astype("int")


### 1.4 Handling imbalance ###
# dividing test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Original Accuracy : ", clf.score(X_test, y_test))

# Print confusion matrix
y_pred = clf.predict(X_test)
cf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("Confusion matrix:\n", cf_mat)

sm = SMOTE(random_state=0)
X_train, y_train = sm.fit_sample(X_train, y_train)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Accuracy after balancing : ", clf.score(X_test, y_test))

# As we are doing cross validation, Applyting SMOTE for complete dataset
sm = SMOTE(random_state=0)
X, y = sm.fit_sample(encoded_data, y)


#### Feature selection ####

# 1. Univariate Feature Selection
# 1.1 Select K best
# top_features = int(0.8 * len(X))
# X = SelectKBest(chi2, k=30).fit_transform(X, y)

# 1.2 Select percentile
# X = SelectPercentile(chi2, percentile=70).fit_transform(X, y)

# 2. Low variance removal
# thresholdVal = 0.8 * (1 - 0.8)
# sel = VarianceThreshold(threshold=(thresholdVal))
# sel.fit_transform(X)

# 3. Greedy Feature Selection
# extraTree = ExtraTreesClassifier()
# scores = model_selection.cross_val_score(extraTree, X, y, cv=10)
# print("Initial Accuracy with Train Test split: ", scores.mean())

# estimator = LogisticRegression(multi_class="auto", solver="lbfgs")
# rfecv = RFECV(estimator, cv=10)
# rfecv.fit(X, y)

# # select highest ranked features and build a new model
# X = X[:, rfecv.support_]

# extraTree = ExtraTreesClassifier()
# scores = model_selection.cross_val_score(extraTree, X, y, cv=10)
# print("Accuracy after feature selection : ", scores.mean())

# 4. correlation matrix with heatmap
# corrmat = pd.DataFrame(X).corr()
# sns.heatmap(corrmat, cmap=plt.cm.Reds)
# plt.show()

# cor_target = y
# imp_features = cor_target[cor_target > 0.5]
# print(imp_features)

### 1.6 Dimension reduction ###
# NCA
# nca = NeighborhoodComponentsAnalysis()
# nca.fit_transform(X, y)

# PCA
pca = PCA().fit(X)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Variance (%)")
plt.title("Variance VS Components")
plt.show()

# Selecting the ideal number of components and fitting the data
pca = PCA(n_components=35)
X = pca.fit_transform(X)

### Training the models ###
models = [
    ("Gaussian NB", GaussianNB()),
    ("KNN", KNeighborsClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("Logistic Regression", LogisticRegression()),
    ("LDA", LinearDiscriminantAnalysis()),
    ("AdaBoost", AdaBoostClassifier()),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Neural Net", MLPClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Extra Trees", ExtraTreesClassifier()),
    # ("SVM", SVC(kernel="linear")),
    ("XGBOOST Classifer", XGBClassifier()),
]

## Model comparison ###
start = timeit.default_timer()

accuracies = []
for name, model in models:

    # kfold = model_selection.KFold(n_splits=10)

    cv_results = model_selection.cross_val_score(model, X, y, cv=5)
    precision = cross_val_score(model, X, y, cv=5, scoring="precision")
    recall = cross_val_score(model, X, y, cv=5, scoring="recall")
    f1 = cross_val_score(model, X, y, cv=5, scoring="f1")

    print(
        "\n ### Classifier :",
        name,
        " ###",
        "\nAccuracy :",
        cv_results.mean(),
        "\nprecision :",
        precision.mean(),
        "\nRecall :",
        recall.mean(),
        "\nF1 Score :",
        f1.mean(),
    )

    accuracies.append((name, cv_results.mean()))


top_3 = nlargest(3, accuracies, key=itemgetter(1))
print("\nTop 3 Accuracies : ", top_3)

stop = timeit.default_timer()
total_time = (stop - start) / 60
print("\nTime taken : ", total_time)

# Graph for the algorithm comparison
plt.bar(range(len(accuracies)), [val[1] for val in accuracies], align="center")
plt.xticks(range(len(accuracies)), [val[0] for val in accuracies])
plt.xticks(rotation=70)
plt.show()

### Hyper-parameter Tuning ###

# paramgrid_ETC = {
#     # "loss": ["deviance"],
#     # "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     # "min_samples_split": list(range(1, 10)),
#     # "min_samples_leaf": list(range(1, 10)),
#     "max_depth": [30, 35, 40],
#     "max_features": [None, "log2", "sqrt"],
#     # "criterion": ["friedman_mse", "mae"],
#     "n_estimators": list(range(100, 600, 100)),
# }
# clf = GridSearchCV(ExtraTreesClassifier(), paramgrid_ETC, cv=10, n_jobs=-1)
# search = clf.fit(X, y)
# print("\n Best Score for Extra Trees :", search.best_score_)
# print("\n Best parameters for Extra Trees :", search.best_params_)

# paramgrid_RFC = {
#     # "loss": ["deviance"],
#     # "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     # "min_samples_split": list(range(1, 10)),
#     # "min_samples_leaf": list(range(1, 10)),
#     "max_depth": list(range(50, 65, 3)),
#     "max_features": [None, "log2", "sqrt"],
#     # "criterion": ["friedman_mse", "mae"],
#     "n_estimators": list(range(175, 225, 10)),
# }
# clf = GridSearchCV(RandomForestClassifier(), paramgrid_RFC, cv=10, n_jobs=-1)
# search = clf.fit(X, y)
# print("\n Best Score for RFC :", search.best_score_)
# print("\n Best parameters for RFC :", search.best_params_)


# paramgrid_KNN = {
#     # "algorithm": "auto",
#     "n_neighbors": list(range(1, 30)),
#     "weights": ["uniform", "distance"],
# }
# clf = GridSearchCV(
#     KNeighborsClassifier(), paramgrid_KNN, cv=10, n_jobs=-1, scoring="accuracy"
# )
# search = clf.fit(X, y)
# print("\n Best Score for KNN :", search.best_score_)
# print("\n Best parameters for KNN :", search.best_params_)

### Testing the paramters on top 3 models ###
optimized_models = [
    (
        "Extra Trees",
        ExtraTreesClassifier(n_estimators=300, max_features="sqrt", max_depth=40),
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=195, max_features="sqrt", max_depth=56),
    ),
    ("KNN", KNeighborsClassifier(weights="uniform", n_neighbors=1)),
]

for name, model in optimized_models:
    cv_results = model_selection.cross_val_score(model, X, y, cv=5)
    precision = cross_val_score(model, X, y, cv=5, scoring="precision")
    recall = cross_val_score(model, X, y, cv=5, scoring="recall")
    f1 = cross_val_score(model, X, y, cv=5, scoring="f1")

    print(
        "\n ### Classifier :",
        name,
        " ###",
        "\nAccuracy :",
        cv_results.mean(),
        "\nprecision :",
        precision.mean(),
        "\nRecall :",
        recall.mean(),
        "\nF1 Score :",
        f1.mean(),
    )
