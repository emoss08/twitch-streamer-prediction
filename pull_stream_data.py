import requests
import json
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

access_token = os.getenv("ACCESS_TOKEN")
client_id = os.getenv("CLIENT_ID")


def pull_streamer_details(streamer_name: str) -> None:
    user_id = ""
    headers = {"Client-ID": f"{client_id}", "Authorization": f"Bearer {access_token}"}
    url = f"https://api.twitch.tv/helix/users?login={streamer_name}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        user_id = data["data"][0]["id"]
    else:
        print(f"Error: {response.status_code}")

    url = (
        f"https://api.twitch.tv/helix/videos?user_id={user_id}&period=all&type=archive"
    )
    headers = {"Client-ID": f"{client_id}", "Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response_json = json.loads(response.text)
    videos = response_json["data"]
    pagination = response_json["pagination"]

    while "cursor" in pagination:
        cursor = pagination["cursor"]
        url = f"https://api.twitch.tv/helix/videos?user_id={user_id}&period=all&type=archive&after={cursor}"
        response = requests.get(url, headers=headers)
        response_json = json.loads(response.text)
        videos += response_json["data"]
        pagination = response_json["pagination"]

    with open("archived_videos.json", "w") as f:
        json.dump(videos, f, indent=4)

    data = pd.read_json("archived_videos.json")

    data["created_at"] = pd.to_datetime(data["created_at"])

    data["day_of_week"] = data["created_at"].dt.dayofweek
    data["hour_of_day"] = data["created_at"].dt.hour
    data["minute_of_hour"] = data["created_at"].dt.minute

    data["streamed_on"] = data["created_at"].notnull().astype(int)

    data = data.drop(
        [
            "id",
            "stream_id",
            "user_id",
            "user_login",
            "user_name",
            "title",
            "description",
            "published_at",
            "url",
            "thumbnail_url",
            "viewable",
            "view_count",
            "language",
            "type",
            "duration",
            "muted_segments",
        ],
        axis=1,
    )

    data["created_at"] = pd.to_datetime(data["created_at"])

    data.to_csv("stream_days.csv", index=False)

    df = pd.read_csv("stream_days.csv")

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["created_at"] = df["created_at"].astype(str)

    df.sort_values("created_at", inplace=True)

    new_df = pd.DataFrame(columns=["created_at", "streamed_on"])

    start_date = df.iloc[0]["created_at"]
    end_date = df.iloc[-1]["created_at"]

    date_range = pd.date_range(start_date, end_date, freq="D")

    for date in date_range:
        if not df["created_at"].str.contains(str(date.date())).any():
            new_df = new_df.append(
                {"created_at": date, "streamed_on": 0}, ignore_index=True
            )

    new_df.to_csv("not_stream_days.csv", index=False)
