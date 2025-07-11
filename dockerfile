FROM python:3.10

WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt

# Download necessary NLTK resources
RUN python -m nltk.downloader stopwords wordnet

# Copy the full app
COPY flask_app/ /app/flask_app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Set Python path so flask_app can find other modules
ENV PYTHONPATH="/app"

EXPOSE 5000

# Run the app inside flask_app folder
#CMD ["python", "flask_app/app.py"]
CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app.app:app"]