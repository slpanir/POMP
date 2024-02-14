import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# from utils import evaluate_from_file
def send_scoring_request(translation_items):
    # url = 'http://xxxx/xcomet_score'
    url = 'http://xxxx/xcomet_score'
    headers = {'Content-Type': 'application/json'}
    session = requests.Session()
    retries = Retry(total=3,  # 总重试次数
                    backoff_factor=1,  # 重试间隔时间的因子
                    status_forcelist=[104, 500, 502, 503, 504])  # 指定哪些状态码需要重试
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        try:
            response = session.post(url, json=translation_items, headers=headers, timeout=300, verify=False)
            if response.status_code == 200:
                return response.json().get('scores', [])
            else:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                if attempts >= max_attempts:
                    print(f"Max attempts reached. Returning score 0.0.")
                    return 0.0
                attempts += 1
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {e}")
            if attempts >= max_attempts:
                print(f"Max attempts reached. Returning score 0.0.")
                return 0.0
            attempts += 1


# 使用示例machine translated text 1
# translation_items = [
#     {"src": "source text 1", "mt": "//////", "ref": "Ғажайып құлаққап."},
#     {"src": "source text 1", "mt": "//////", },
# ]
# #
# result = send_scoring_request(translation_items)
# print(result)

# results = evaluate_from_file("chatgpt-super_sample_got_test_v2_2024-02-01_03-14-00", "et")
# print(results)
