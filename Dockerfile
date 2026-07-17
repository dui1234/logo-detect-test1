# Official Ultralytics GPU-capable image
FROM ultralytics/ultralytics:latest


# Set the working directory inside the container
WORKDIR /workspace


# Make Python output appear immediately in Docker logs
ENV PYTHONUNBUFFERED=1


# Copy the YOLO11s pretrained model
COPY yolo11s.pt /workspace/yolo11s.pt


# Copy the training script
COPY train_main.py /workspace/train_main.py


# Copy the complete dataset
COPY bank_logo_dataset /workspace/bank_logo_dataset

# The Python script will always be executed
ENTRYPOINT ["python", "/workspace/train_main.py"]


# Default arguments
CMD ["--checkpoint-dir", "/checkpoints/default_run", "--epochs", "150", "--imgsz", "640", "--device", "0", "--save-period", "10"]
