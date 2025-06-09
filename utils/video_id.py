from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if 'youtube' in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path.strip('/')
    return None