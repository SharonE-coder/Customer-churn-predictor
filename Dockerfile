# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt -i https://pypi.org/simple

# Expose Streamlit default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
