from src.yolov7 import YOLOv7
from src.yolov5 import YOLOv5
from src.logger import Logger

if __name__ == "__main__":

    yolov7_model = YOLOv7(verbosity=3, yolo_repo_download_path="E:/projetos/UFSC/YOLOVerse/yolo_repo")
    yolov7_model.train(
        project_name='trainings',
        run_name='v7_train',
        weights='D:/downloads/yolov7.pt',
        data='D:/downloads/Basketball Players.v22-raw-images-scoreboardclassesonly-nonulls.yolov7pytorch/data.yaml',
        batch_size=8,
        num_epochs=4,
        device=0,
    )

    yolov5_model = YOLOv5(verbosity=3, yolo_repo_download_path="E:/projetos/UFSC/YOLOVerse/yolo_repo")
    yolov5_model.train(
        project_name='trainings',
        run_name='v5_train',
        weights='D:/downloads/yolov5s.pt',
        data='D:/downloads/Basketball Players.v22-raw-images-scoreboardclassesonly-nonulls.yolov7pytorch/data.yaml',
        batch_size=8,
        num_epochs=4,
        device=0,
    )

    
