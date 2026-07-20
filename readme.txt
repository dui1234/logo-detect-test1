https://www.dropbox.com/scl/fi/8jpjaua9oh8xkxnsl429j/bank_logo_dataset.zip?rlkey=rdy03lfrrscrg6b3cf7lkcsmo&st=u6y1d458&dl=0


SKIP_PREDICTIONS=1 ./run_yolo_test.sh \
    ./bank_logo_dataset \
    ./yolo11s_640_exp01/weights/best.pt \
    ./yolo11s_960_exp02/weights/best.pt \
    ./yolo_fps_results_cpu_b1 \
    cpu \
    4 \
    1 \
    5 \
    0.50 \
    0.45 \
    4 \
    2


20/7/26
time yolo detect predict model=../yolo11s_640_exp01/weights/best.pt source=../bank_logo_dataset/images/test_0.5 imgsz=640 conf=0.50 iou=0.45 device=cpu save=True save_txt=True save_conf=True project=./pred_results_2 name=yolo11_640

- bash: |
    echo "Removing local evaluation Docker images..."

    docker rmi -f \
      "$(REGISTRY_JFROG_HOST)/$(yoloEvalImage):$(IMAGE_TAG)" \
      || true

    docker rmi -f \
      "$(REGISTRY_JFROG_HOST)/$(yoloEvalImage):build-$(Build.BuildId)" \
      || true

    echo "Local Docker image cleanup completed."
  displayName: Clean Up Local Docker Images

CMD ["--photos-root", "/workspace/Photos", "--labels-root", "/workspace/labels/test", "--dataset-yaml", "/workspace/dataset.yaml", "--model-640", "/workspace/models/yolo11s_640_best.pt", "--model-960", "/workspace/models/yolo11s_960_best.pt", "--output-dir", "/outputs", "--device", "0", "--val-batch", "16", "--benchmark-batch", "1", "--predict-batch", "16", "--benchmark-repeats", "5", "--warmup-runs", "3", "--workers", "8", "--val-conf", "0.001", "--val-nms-iou", "0.70", "--predict-conf", "0.50", "--predict-iou", "0.45", "--half"]
