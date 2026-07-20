from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_invalid_feature_count():
    response = client.post(
        "/predict",
        json={
            "student_id": "TEST-ERROR",
            "email_tutor": "tutor@universidad.local",
            "features": [1.0, 2.0, 3.0],
        },
    )

    # Si el modelo aún no se cargó devuelve 503; con el modelo devuelve 422.
    assert response.status_code in {422, 503}
