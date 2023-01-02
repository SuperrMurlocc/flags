import os
import cv2

COUNTRIES_PATH = 'flags'


def get_countries() -> list[str]:
    return [country for country in os.listdir(COUNTRIES_PATH)]


def get_representatives() -> dict:
    res = {}
    for country in os.listdir(COUNTRIES_PATH):
        if country == '.DS_Store':
            continue
        res[country] = cv2.cvtColor(cv2.imread(f"{COUNTRIES_PATH}/{country}/{list(filter(lambda file: 'representative' in file, os.listdir(f'{COUNTRIES_PATH}/{country}')))[0]}"), cv2.COLOR_RGB2BGR)
    return res


if __name__ == '__main__':
    print(get_representatives())
