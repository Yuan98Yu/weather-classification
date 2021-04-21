import requests

default_cfg = {
    'url': 'https://devapi.qweather.com/v7/weather/now?',
    'parameters_of_get': {
        'location': '101010100',
        'key': 'a6ac911ef44645eb830ca938f1911fdb'
    }
}


def get_weather_info(cfg=default_cfg):
    url = cfg['url']
    parameters_of_get = cfg['parameters_of_get']
    try:
        res = requests.get(url=url, params=parameters_of_get)
        res.raise_for_status()
        # res.encoding = res.apparent_encoding
        weather_info = res.json()['now']
        weather_info['type'] = weather_info.pop('text')
        # print(type(weather_info))
        return weather_info
    except Exception as e:
        print(e)


if __name__ == '__main__':
    print(get_weather_info())
