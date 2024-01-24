import requests

def send_scoring_request(translation_items):
    url = ''
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=translation_items, headers=headers)
    print(response.status_code)
    print(response.text)
    return response.json()

# 使用示例
translation_items = [
    {"src": "source text 1", "mt": "machine translated text 1", "hyp": "hypothesis text 1"}
]

result = send_scoring_request(translation_items)
print(result)
