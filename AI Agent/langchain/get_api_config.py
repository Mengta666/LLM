from configparser import ConfigParser

class GetApiConfig:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read('../config/all_apis.ini')

    def get_baidu_weather_api_key(self):
        return self.config['百度天气AK']['YOUR_AK']

    def get_gemini_api_key(self, api_number:int):
        return self.config['gemini-api'][ f'gemini_api_key_{api_number}']