"""
Production deployment tests for AI Pipeline API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from datetime import datetime

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from deployment import app, model_registry, prediction_service


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def valid_api_key():
    """Valid API key for testing"""
    return "test-api-key-12345"

@pytest.fixture
def patient_data():
    """Sample patient data for testing"""
    return {
        "patient_id": "patient-123",
        "biomarkers": {
            "marker1": 2.5,
            "marker2": 1.8,
            "marker3": 3.2
        },
        "clinical_data": {
            "age": 45,
            "gender": "M"
        },
        "demographics": {
            "weight": 75,
            "height": 180
        }
    }

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert "dependencies" in data
        assert "metrics" in data

    def test_health_check_response_format(self, client):
        """Test health check response format"""
        response = client.get("/health")
        data = response.json()
        
        # Check required fields
        required_fields = ["status", "timestamp", "version", "uptime", "dependencies", "metrics"]
        for field in required_fields:
            assert field in data
        
        # Check dependencies format
        assert isinstance(data["dependencies"], dict)
        assert "database" in data["dependencies"]
        assert "model_registry" in data["dependencies"]


class TestPredictionEndpoint:
    """Test prediction endpoint"""
    
    def test_prediction_unauthorized(self, client, patient_data):
        """Test prediction without authorization"""
        prediction_request = {
            "patient_data": patient_data,
            "model_type": "ensemble",
            "include_confidence": True
        }
        
        response = client.post("/predict", json=prediction_request)
        assert response.status_code == 403  # Forbidden due to missing auth

    def test_prediction_with_auth(self, client, patient_data, valid_api_key):
        """Test prediction with valid authorization"""
        prediction_request = {
            "patient_data": patient_data,
            "model_type": "ensemble",
            "include_confidence": True
        }
        
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.post("/predict", json=prediction_request, headers=headers)
        
        # Should succeed with valid auth
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction_id" in data
        assert "patient_id" in data
        assert "risk_score" in data
        assert "risk_category" in data
        assert "confidence_score" in data
        assert "biomarker_importance" in data
        assert "recommendations" in data

    def test_prediction_response_format(self, client, patient_data, valid_api_key):
        """Test prediction response format"""
        prediction_request = {
            "patient_data": patient_data
        }
        
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.post("/predict", json=prediction_request, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check data types
        assert isinstance(data["risk_score"], float)
        assert 0 <= data["risk_score"] <= 1
        assert data["risk_category"] in ["low", "moderate", "high"]
        assert isinstance(data["confidence_score"], float)
        assert 0 <= data["confidence_score"] <= 1
        assert isinstance(data["biomarker_importance"], dict)
        assert isinstance(data["recommendations"], list)

    def test_prediction_invalid_data(self, client, valid_api_key):
        """Test prediction with invalid data"""
        invalid_request = {
            "patient_data": {
                "patient_id": "test",
                "biomarkers": {}  # Empty biomarkers should fail validation
            }
        }
        
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.post("/predict", json=invalid_request, headers=headers)
        
        assert response.status_code == 422  # Validation error


class TestModelEndpoints:
    """Test model management endpoints"""
    
    def test_list_models(self, client, valid_api_key):
        """Test listing available models"""
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.get("/models", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_activate_model_success(self, client, valid_api_key):
        """Test successful model activation"""
        # First, ensure we have a model to activate
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        
        # Get available models
        models_response = client.get("/models", headers=headers)
        models = models_response.json()["models"]
        
        if models:
            model_name = models[0]["name"]
            response = client.post(f"/models/{model_name}/activate", headers=headers)
            assert response.status_code == 200

    def test_activate_nonexistent_model(self, client, valid_api_key):
        """Test activating non-existent model"""
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.post("/models/nonexistent/activate", headers=headers)
        
        assert response.status_code == 404


class TestMonitoringEndpoints:
    """Test monitoring endpoints"""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_alerts_endpoint(self, client, valid_api_key):
        """Test alerts endpoint"""
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.get("/monitoring/alerts", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "system_metrics" in data
        assert "alert_count" in data

    def test_system_status_endpoint(self, client, valid_api_key):
        """Test system status endpoint"""
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.get("/monitoring/system", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "system_metrics" in data
        assert "service_status" in data


class TestPredictionHistory:
    """Test prediction history endpoints"""
    
    def test_list_predictions_unauthorized(self, client):
        """Test listing predictions without auth"""
        response = client.get("/predictions")
        assert response.status_code == 403

    def test_list_predictions_authorized(self, client, valid_api_key):
        """Test listing predictions with auth"""
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.get("/predictions", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_get_prediction_by_id_not_found(self, client, valid_api_key):
        """Test getting non-existent prediction"""
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.get("/predictions/nonexistent-id", headers=headers)
        
        assert response.status_code == 404


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @patch('src.deployment.security_manager.check_rate_limit')
    def test_rate_limit_exceeded(self, mock_rate_limit, client, patient_data, valid_api_key):
        """Test rate limiting enforcement"""
        mock_rate_limit.return_value = False  # Simulate rate limit exceeded
        
        prediction_request = {
            "patient_data": patient_data
        }
        
        headers = {"Authorization": f"Bearer {valid_api_key}"}
        response = client.post("/predict", json=prediction_request, headers=headers)
        
        assert response.status_code == 429  # Too Many Requests


class TestSecurityFeatures:
    """Test security features"""
    
    def test_invalid_api_key(self, client, patient_data):
        """Test request with invalid API key"""
        prediction_request = {
            "patient_data": patient_data
        }
        
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.post("/predict", json=prediction_request, headers=headers)
        
        assert response.status_code == 401  # Unauthorized

    def test_missing_authorization_header(self, client, patient_data):
        """Test request without authorization header"""
        prediction_request = {
            "patient_data": patient_data
        }
        
        response = client.post("/predict", json=prediction_request)
        assert response.status_code == 403  # Forbidden


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations"""
    
    async def test_prediction_service_async(self, patient_data):
        """Test async prediction service"""
        from deployment import PredictionRequest, PatientData
        
        # Create request
        patient_data_obj = PatientData(**patient_data)
        request = PredictionRequest(patient_data=patient_data_obj)
        
        # Mock model registry
        with patch.object(model_registry, 'get_active_model', return_value=Mock()):
            response = await prediction_service.make_prediction(request)
            
            assert response.patient_id == patient_data["patient_id"]
            assert 0 <= response.risk_score <= 1
            assert response.risk_category in ["low", "moderate", "high"]


class TestDataValidation:
    """Test input data validation"""
    
    def test_patient_data_validation(self):
        """Test patient data model validation"""
        from deployment import PatientData
        
        # Valid data
        valid_data = {
            "patient_id": "test-123",
            "biomarkers": {"marker1": 1.5}
        }
        patient = PatientData(**valid_data)
        assert patient.patient_id == "test-123"
        
        # Invalid data - empty biomarkers
        with pytest.raises(ValueError):
            PatientData(patient_id="test", biomarkers={})

    def test_prediction_request_validation(self):
        """Test prediction request validation"""
        from deployment import PredictionRequest, PatientData
        
        patient_data = PatientData(
            patient_id="test",
            biomarkers={"marker1": 1.0}
        )
        
        # Valid request
        request = PredictionRequest(patient_data=patient_data)
        assert request.model_type == "ensemble"  # Default value
        assert request.include_confidence is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
