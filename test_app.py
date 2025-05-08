import pytest
from fastapi.testclient import TestClient
from app import app  # Ensure app.py and test_app.py are in the same folder

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_predict_valid_input():
    payload = {"sentence": "The company did not disclose its liabilities properly."}
    response = client.post("/predict_sync", json=payload)
    assert response.status_code == 200
    # Expect a 'label' key in response
    assert "label" in response.json()

def test_predict_empty_input():
    payload = {"sentence": ""}
    response = client.post("/predict_sync", json=payload)
    assert response.status_code == 200  # Even empty input should return a prediction
    assert "label" in response.json()

if __name__ == "__main__":
    pytest.main()