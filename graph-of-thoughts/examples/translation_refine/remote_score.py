import requests

def send_scoring_request(translation_items):
    url = 'http://10.107.18.40:8080/xcomet_score'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=translation_items, headers=headers)
    # print(response.status_code)
    # print(response.text)
    return response.json()['scores']

# 使用示例machine translated text 1
translation_items = [
    {"src": "source text 1", "mt": "//////", "ref": "Ғажайып құлаққап."}
]

result = send_scoring_request(translation_items)
print(result)
