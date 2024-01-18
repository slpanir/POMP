import json
import random
import re

import pandas as pd
import openai

# 载入你的OpenAI API秘钥
with open("key.txt", 'r', encoding='utf-8') as f:
    keys = [i.strip() for i in f.readlines()]


def generate(prompt):
    index_num = random.randint(0, len(keys) - 1)
    openai.api_key = keys[index_num]  # 每次调用使用不同的key防止被ban

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "你是一个精通处理新闻的助手。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    generation = response['choices'][0]['message']['content']
    print(generation)

    return generation

def get_simple_news():
    error_count = 0

    with open("../data/summaries.json", 'r', encoding='utf8') as file:
        summaries = json.load(file)

    try:
        with open("../data/simple_news.json", 'r', encoding='utf8') as file:
            simple_news = json.load(file)
        count = len(simple_news)
    except:
        simple_news = {}
        count = 0

    print(len(summaries))
    while count < len(summaries.keys()):
        for index in summaries:
            if str(index) in simple_news.keys():
                count += 1
                print(index, "done")
                continue
            try:
                prompt_4 = "根据以下内容完成一篇新闻报道，首先生成新闻标题，换行后生成新闻内容：\n" +summaries[index] + "\n"
                prompt_4 = prompt_4[:2500]
                news = generate(prompt_4)
                simple_news[index] = news

            except Exception as e:
                print("ID:", index)
                print(str(e))
                error_count += 1
                if error_count > 200:
                    exit()
            with open("../data/simple_news.json", 'w', encoding='utf8') as json_file:
                json.dump(simple_news, json_file, ensure_ascii=False)

if __name__ == '__main__':

    get_simple_news()
