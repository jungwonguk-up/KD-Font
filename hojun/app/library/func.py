import base64
import json
import click
import requests



def request_rest(id: str, image: bytes):
    # serving_address = None #TODO
    
    headers = {"Content-Type": "application/json"}
    base64_image = base64.urlsafe_b64encode(image).decode("ascii")
    request_dict = {"inputs": {"image": [base64_image]}}
    response = requests.post(
        # serving_address,
        json.dumps(request_dict),
        headers=headers,
    )
    return dict(response.json())['outputs'][0]


