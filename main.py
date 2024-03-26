from src.yolov7 import YOLOv7
from src.logger import Logger

my_model = YOLOv7(verbosity=3, yolo_repo_download_path="yolo_repo", force_gpu=True)
my_model.train(
    project_name='trainings',
    run_name='train-',
    start_weights_path='D:\downloads\yolov7.pt',
    data_yaml_path='D:/downloads/Basketball Players.v22-raw-images-scoreboardclassesonly-nonulls.yolov7pytorch/data.yaml',
    batch_size=8,
    num_epochs=100
)
