from google import genai
import configparser

config = configparser.ConfigParser()
config.read("./AI Agent/config/all_apis.ini")
api_key = config["gemini-api"]["gemini_api_key_1"]
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="models/gemini-2.5-flash", contents="中文回答问题，告诉我当前我所在的IP，并且告诉我当地的天气"
)
print(response.text)