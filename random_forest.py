################################
# training has been done in Colab
################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

x= y=x_train= x_test= y_train= y_test=rf_model=y_pred=""


################################
# to load data and split them
################################
def call_dataset():

    # laod data
    df = pd.read_csv("PC_SYM_2.csv")

    # split into x(features) and y(classes)
    y = df.iloc[:, 0] 
    x = df.iloc[:, 1:]

    # train and test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,  # train 80 - test 20
        random_state=41,
        shuffle=True
    )

    # train=212     test=57
    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)


################################
# to prepare model and run it
################################
def train_model():

    # prepare model
    # n_estimators (creating 300 decision trees)
    rf_model = RandomForestClassifier(n_estimators=300, random_state=41)

    # train model
    rf_model.fit(x_train, y_train)

    # predictions on test data
    y_pred = rf_model.predict(x_test)


################################
# to get the performance
################################
def calc_acc():

    # evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # detailed report for percision, recall, f1-score
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique()))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


################################
# to understand what features have more influnce
################################
def get_feature_importance():
    
    # get feature importance
    importances = rf_model.feature_importances_
    feature_names = x.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # plot
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (RandomForest)")
    plt.xlabel("Importance Score")
    plt.show()

    print("\nFEATURE IMPORTANCE:")
    print(importance_df)


