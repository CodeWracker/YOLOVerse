from src.yolov7 import YOLOv7
from src.yolov5 import YOLOv5
from src.logger import Logger

if __name__ == "__main__":
    my_model = YOLOv5(verbosity=3, yolo_repo_download_path="yolo_repo")
    my_model.train(
        project_name='trainings',
        run_name='train-n',
        weights='D:\downloads\yolov5s.pt',
        data='D:/downloads/Basketball Players.v22-raw-images-scoreboardclassesonly-nonulls.yolov7pytorch/data.yaml',
        batch_size=8,
        num_epochs=4,
        device=0,
    )
