from livekit import api
import os
from dotenv import load_dotenv

load_dotenv()

def create_token(identity: str, room: str):

    at = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET"),
    )

    at = at.with_identity(identity)
    at = at.with_name(identity)
    at = at.with_grants(
        api.VideoGrants(
            room_join=True,
            room=room,
        )
    )

    return at.to_jwt()
