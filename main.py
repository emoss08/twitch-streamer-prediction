import pandas as pd
import numpy as np
from pandas import Timestamp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple
import time
import pull_stream_data
import warnings

warnings.filterwarnings("ignore")


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


def main(stream_date: str, streamer_name: str) -> None:
    pull_stream_data.pull_streamer_details(streamer_name=streamer_name)

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

    prediction_date: Timestamp = pd.Timestamp(stream_date)
    prediction_features = [[prediction_date.timestamp(), prediction_date.dayofweek]]
    scaled_prediction_features = scaler.transform(prediction_features)
    prediction = clf.predict(scaled_prediction_features)[0]
    prediction_proba = clf.predict_proba(scaled_prediction_features)[0]

    if prediction == 1:
        print(
            f"{streamer_name} will stream on {stream_date} with a probability of {prediction_proba[1]:.2f}\n"
            "must be nice."
        )
    else:
        print(
            f"{streamer_name} will not stream on {stream_date} with a probability of {prediction_proba[0]:.2f}\n"
            "Don't act surprised."
        )


if __name__ == "__main__":
    streamer_name = input("Give the username of the streamer who is full of shit...? ")
    print("hmmmm.... interesting. I hope they ban you.")
    time.sleep(1)
    print("anyways..... Moving on.....")
    print(
        "I'm going to ask for a date next, type it in correctly.... FFS in this format: (2022-03-29)"
    )
    time.sleep(1)
    stream_date = input("Tell me what date they said they were streaming? ")
    print("smh, what a waste of time... The answer is no. Actually idk. Moving on....")
    time.sleep(1)
    main(stream_date=stream_date, streamer_name=streamer_name)
