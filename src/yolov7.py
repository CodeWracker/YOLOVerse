# THIS FILE IS TO ENCAPSULATE THE USAGE OF THE YOLOv7 API CREATED BY @WonkKinYiu (https://github.com/WongKinYiu/yolov7)

import sys
from pathlib import Path
# add to path the location of this file
YOLOV7_FILE = Path(__file__).resolve()
ROOT_YOLOV7_FILE = YOLOV7_FILE.parents[0]
if str(ROOT_YOLOV7_FILE) not in sys.path:
    sys.path.append(str(ROOT_YOLOV7_FILE))



from yolo import YOLO, BoundingBoxDetection


class YOLOv7(YOLO):
    """This class is a wrapper over the YOLOv7 Implementation, standardizing the API."""

    def __init__(self, yolo_repo_download_path: str = "yolo_repo", verbosity:int = 0) -> None:
        super().__init__(yolo_repo_download_path=yolo_repo_download_path, verbosity=verbosity)


        # donwload the repo git git clone and delete the .git folder
        self._download_repo('yolov7')


    # override the method to train
    def train(self, project_name: str, run_name: str, start_weights_path: str, data_yaml_path: str, batch_size: int, num_epochs: int) -> None:
        return  