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
