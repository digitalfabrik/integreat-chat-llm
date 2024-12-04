import json
from urllib.request import urlopen
from urllib.parse import quote

from django.conf import settings


def translate_source_path(path: str, wanted_language: str) -> str:
    """
    Get the page path for a specified language
    """
    region = path.split("/")[1]
    cur_language = path.split("/")[2]
    pages_url = (
        f"https://{settings.INTEGREAT_CMS_DOMAIN}/api/v3/{region}/"
        f"{cur_language}/children/?url={path}&depth=0"
    )
    encoded_url = quote(pages_url, safe=':/=?&')
    response = urlopen(encoded_url)
    return json.loads(response.read())[0]["available_languages"][wanted_language]["path"]
