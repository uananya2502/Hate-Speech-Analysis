from googleapiclient.discovery import build
import config

youtube = build(
    "youtube",
    "v3",
    developerKey=config.API_KEY
)

request = youtube.search().list(
    q="VIT placement",
    part="snippet",
    type="video",
    maxResults=3
)

response = request.execute()

for item in response["items"]:
    print(item["snippet"]["title"])