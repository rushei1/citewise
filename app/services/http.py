import requests


class HttpError(Exception):
    """An external HTTP call failed."""


class Http:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise HttpError(str(e)) from e
        if res.status_code != 200:
            raise HttpError(f"HTTP {res.status_code} from {url}")
        return res.json()
