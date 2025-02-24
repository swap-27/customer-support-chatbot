# Use the latest stable Python image (modify version if needed)
FROM python:3.10.13

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the project directory into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 for the app to run
EXPOSE 7860

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]