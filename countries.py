import os


def get_countries(path) -> list[str]:
    return [country for country in os.listdir(path)]
