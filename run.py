"""
Entry point — generates data, seeds DB, trains models, and starts the server.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup():
    """Run all setup steps: generate data, seed DB, train models."""
    
    data_dir = PROJECT_ROOT / "data" / "generated"
    db_path = PROJECT_ROOT / "app.db"
    model_dir = PROJECT_ROOT / "models"
    
    # Step 1: Generate synthetic dataset
    if not data_dir.exists():
        print("\n" + "="*60)
        print("📊 STEP 1: Generating synthetic dataset...")
        print("="*60)
        from data.generate_dataset import main as generate_data
        generate_data()
    else:
        print("✅ Dataset already exists, skipping generation")
    
    # Step 2: Seed the database
    if not db_path.exists():
        print("\n" + "="*60)
        print("📦 STEP 2: Seeding database...")
        print("="*60)
        from data.seed_db import seed_database
        seed_database()
    else:
        print("✅ Database already exists, skipping seeding")
    
    # Step 3: Train ML models
    if not model_dir.exists() or not list(model_dir.glob("*.joblib")):
        print("\n" + "="*60)
        print("🧠 STEP 3: Training ML models...")
        print("="*60)
        from ml.trainer import train_models
        train_models()
    else:
        print("✅ Trained models found, skipping training")


def start_server():
    """Start the FastAPI server."""
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 STEP 4: Starting RecoAI server...")
    print("="*60)
    port = int(os.environ.get("PORT", 8000))
    print(f"   URL: http://localhost:{port}")
    print(f"   API Docs: http://localhost:{port}/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    os.chdir(str(PROJECT_ROOT))
    setup()
    start_server()
