# Use an official Python runtime as the base image
FROM python:3.11.2

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libxcb-xinerama0 libxcb-xkb1 libxcb-icccm4
RUN apt-get install -y libpulse-mainloop-glib0

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Copy the Python script into the container
COPY . .

# Set the command to run your script
CMD ["python", "football_analysis.py"]
