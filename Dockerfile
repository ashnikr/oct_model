# Use a minimal Python base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy and install Python dependencies
COPY deploy/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your project code
COPY main/ ./main/
COPY model/ ./model/

# Disable Streamlit telemetry
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "main/testt.py", "--server.port=8501", "--server.address=0.0.0.0"]
