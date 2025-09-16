#!/usr/bin/env python3
"""
Demo Launch Script
Automated setup and launch for AI Pipeline federated personalization demo
"""

import subprocess
import time
from pathlib import Path

# Use virtual environment Python
PYTHON_CMD = "/Users/jasoneades/ai-pipeline/.venv/bin/python"


def check_requirements():
    """Check if all required components are available"""

    print("🔍 Checking demo requirements...")

    # Check for enhanced data
    data_dir = Path("data/demo_enhanced")
    if not data_dir.exists():
        print("❌ Enhanced demo data not found")
        print("🔧 Generating enhanced synthetic data...")

        try:
            result = subprocess.run(
                [PYTHON_CMD, "enhanced_synthetic_generator.py"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.returncode == 0:
                print("✅ Enhanced synthetic data generated successfully")
            else:
                print(f"❌ Failed to generate synthetic data: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error generating synthetic data: {e}")
            return False
    else:
        print("✅ Enhanced demo data found")

    # Check for competitive analysis data
    try:
        from presentation.application_competitive_analysis import (
            calculate_application_scores,
        )

        print("✅ Competitive analysis module available")
    except ImportError:
        print("⚠️  Competitive analysis module not found (demo will use fallback)")

    return True


def launch_demo():
    """Launch the Streamlit demo"""

    print("\n🚀 Launching AI Pipeline Demo...")
    print("📱 Demo will open in your default web browser")
    print("🌐 Demo URL: http://localhost:8501")
    print("\n" + "=" * 50)
    print("AI PIPELINE FEDERATED PERSONALIZATION DEMO")
    print("=" * 50)

    try:
        # Launch Streamlit
        subprocess.run(
            [
                PYTHON_CMD,
                "-m",
                "streamlit",
                "run",
                "streamlit_demo.py",
                "--server.headless",
                "false",
                "--server.port",
                "8501",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=".",
        )

    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching demo: {e}")


def main():
    """Main demo launcher"""

    print("=" * 60)
    print("🧬 AI PIPELINE FEDERATED PERSONALIZATION DEMO LAUNCHER")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        print("\n❌ Demo requirements not met. Please resolve issues and try again.")
        return

    print("\n✅ All requirements met")

    # Brief pause for readability
    time.sleep(1)

    # Launch demo
    launch_demo()


if __name__ == "__main__":
    main()
