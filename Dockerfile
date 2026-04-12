FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate data, seed DB, and train models at build time
# This means the container starts instantly with pre-trained models
RUN python -c "import sys; sys.path.insert(0,'.'); from data.generate_dataset import main; main()"
RUN python -c "import sys; sys.path.insert(0,'.'); from data.seed_db import seed_database; seed_database()"
RUN python -c "import sys; sys.path.insert(0,'.'); from ml.trainer import train_models; train_models()"

# Cloud platforms inject PORT env variable
ENV PORT=8000
EXPOSE ${PORT}

CMD ["python", "run.py"]
