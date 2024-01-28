from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_valid_text_input():
    response = client.post("/classify/", json={"text": "geländer biegen"})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_empty_text_input():
    response = client.post("/classify/", json={"text": ""})
    assert response.status_code == 200 

def test_long_text_input():
    long_text = "Ein sehr langer Text in deutscher Sprache, der viele verschiedene Aspekte und Themen abdeckt."
    response = client.post("/classify/", json={"text": long_text})
    assert response.status_code == 200

def test_special_characters_input():
    response = client.post("/classify/", json={"text": "1234567890 !@#$%^&*()"})
    assert response.status_code == 200

def test_invalid_json_format():
    response = client.post("/classify/", data="{\"text: \"malformed json\"}")
    assert response.status_code == 422
