"""Test script for AgroVeyra Backend endpoints."""

import io
import json
import sys

import requests
from PIL import Image

BASE_URL = "http://localhost:8000"


def test_health():
    """Test GET / endpoint."""
    print("\n" + "=" * 50)
    print("TEST 1: GET /")
    print("=" * 50)
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code == 200 and response.json().get("status") == "running":
            print(">>> PASS <<<")
            return True
        else:
            print(">>> FAIL <<<")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(">>> FAIL <<<")
        return False


def test_predict():
    """Test POST /predict endpoint."""
    print("\n" + "=" * 50)
    print("TEST 2: POST /predict")
    print("=" * 50)
    try:
        # Try using existing test_leaf.jpg, otherwise create a dummy image
        try:
            with open("test_leaf.jpg", "rb") as f:
                image_data = f.read()
            print("Using test_leaf.jpg from disk")
        except FileNotFoundError:
            print("test_leaf.jpg not found, creating dummy 224x224 green image")
            img = Image.new("RGB", (224, 224), color=(34, 139, 34))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_data = buf.getvalue()

        files = {"file": ("test_leaf.jpg", image_data, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        if response.status_code == 200 and "disease" in data:
            print(">>> PASS <<<")
            return True
        else:
            print(">>> FAIL <<<")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(">>> FAIL <<<")
        return False


def test_weather():
    """Test POST /weather endpoint."""
    print("\n" + "=" * 50)
    print("TEST 3: POST /weather")
    print("=" * 50)
    try:
        payload = {"lat": 12.9716, "lon": 77.5946, "disease": "Early_blight"}
        response = requests.post(f"{BASE_URL}/weather", json=payload, timeout=15)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        if response.status_code == 200 and "spread_risk" in data:
            print(">>> PASS <<<")
            return True
        else:
            print(">>> FAIL <<<")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(">>> FAIL <<<")
        return False


if __name__ == "__main__":
    import json

    print("AgroVeyra Backend Test Suite")
    print(f"Target: {BASE_URL}")
    print("Make sure the backend is running: uvicorn main:app --reload --port 8000")

    results = []
    results.append(test_health())
    results.append(test_predict())
    results.append(test_weather())

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed. Check the output above.")


