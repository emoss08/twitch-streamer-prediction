import pandas as pd
import numpy as np
from pandas import Timestamp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple

STREAM_DATE = "2022-03-29"
STREAMER_NAME = "Acorn1010"


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray]:
    stream_data: pd.DataFrame = pd.read_csv("stream_days.csv")
    not_stream_data: pd.DataFrame = pd.read_csv("not_stream_days.csv")

    stream_data["label"] = 1
    not_stream_data["label"] = 0

    data: pd.DataFrame = pd.concat([stream_data, not_stream_data])

    data["created_at"] = pd.to_datetime(data["created_at"])
    data["day_of_week"] = data["created_at"].dt.dayofweek
    data["timestamp"] = data["created_at"].apply(lambda x: x.timestamp())

    X = data[["timestamp", "day_of_week"]].to_numpy()
    y = data["label"].to_numpy()

    return X, y


def main() -> None:
    X, y = load_and_preprocess_data()

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print("Cross-validation accuracy: {:.2f}".format(scores.mean()))

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))

    prediction_date: Timestamp = pd.Timestamp(STREAM_DATE)
    prediction_features = [[prediction_date.timestamp(), prediction_date.dayofweek]]
    scaled_prediction_features = scaler.transform(prediction_features)
    prediction = clf.predict(scaled_prediction_features)[0]
    prediction_proba = clf.predict_proba(scaled_prediction_features)[0]

    if prediction == 1:
        print(
            f"{STREAMER_NAME} will stream on {STREAM_DATE} with a probability of {prediction_proba[1]:.2f}"
        )
    else:
        print(
            f"{STREAMER_NAME} will not stream on {STREAM_DATE} with a probability of {prediction_proba[0]:.2f}"
        )


if __name__ == "__main__":
    main()
