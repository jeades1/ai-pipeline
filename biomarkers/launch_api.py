"""
Clinical API Server Launcher

Launch the FastAPI server for the clinical biomarker risk assessment API.
This provides a production-ready web service for real-time clinical decision support.

Author: AI Pipeline Team
Date: September 2025
"""

import uvicorn
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def launch_clinical_api():
    """Launch the clinical API server"""

    print("=" * 70)
    print("🏥 CLINICAL BIOMARKER RISK ASSESSMENT API SERVER")
    print("=" * 70)
    print("🚀 Starting FastAPI server...")
    print("📡 API Documentation will be available at: http://localhost:8000/docs")
    print("📋 Alternative docs at: http://localhost:8000/redoc")
    print("💾 Health check endpoint: http://localhost:8000/health")
    print("🔧 Production-ready with multi-omics integration")
    print("-" * 70)

    # Launch the FastAPI server
    try:
        uvicorn.run(
            "biomarkers.clinical_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")


if __name__ == "__main__":
    launch_clinical_api()
