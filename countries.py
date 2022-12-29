import os

COUNTRIES_PATH = 'flags'


def get_countries() -> list[str]:
    return [country for country in os.listdir(COUNTRIES_PATH)]
