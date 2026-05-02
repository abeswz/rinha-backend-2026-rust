import json

import requests


def process_payload_backend(payload):
    url = "http://localhost:9999/fraud-score"
    r = requests.post(url, json=payload)

    if r.status_code == 200:
        print(f"request sent with success: {r.content}")
    else:
        print(f"error request: {r.status_code} | transaction: {payload['id']}")


def check_backend_health():
    url = "http://localhost:9999/ready"
    r = requests.get(url)

    return r.status_code == 200


def process_all_payloads():
    """
    Process all payloads into the payloads files.
    """
    if not check_backend_health():
        return

    with open("./test/mocks/payloads.json", "r", encoding="utf-8") as payloads:
        data = json.load(payloads)

        for payload in data:
            process_payload_backend(payload)


def main():
    process_all_payloads()


if __name__ == "__main__":
    main()
