# THIS FILE IS TO ENCAPSULATE THE USAGE OF THE YOLOv7 API CREATED BY @WonkKinYiu (https://github.com/WongKinYiu/yolov7)

import sys
from pathlib import Path
import os
import torch
import yaml

# add to path the location of this file
YOLOV7_FILE = Path(__file__).resolve()
ROOT_YOLOV7_FILE = YOLOV7_FILE.parents[0]
if str(ROOT_YOLOV7_FILE) not in sys.path:
    sys.path.append(str(ROOT_YOLOV7_FILE))



from yolo import YOLO, BoundingBoxDetection


class YOLOv7(YOLO):
    """This class is a wrapper over the YOLOv7 Implementation, standardizing the API."""

    def __init__(self, yolo_repo_download_path: str = None, verbosity:int = 0, force_gpu:bool = False) -> None:
        super().__init__(yolo_repo_download_path=yolo_repo_download_path, verbosity=verbosity, force_gpu=force_gpu)


        # donwload the repo git git clone and delete the .git folder
        self._download_repo('yolov7')
        # self._setup()

    # override the method to train
    def train(self, project_name: str, run_name: str, start_weights_path: str, data_yaml_path: str, batch_size: int, num_epochs: int) -> None:
        # import the necessarie dependencies
        self.handle_log_event("Importing the necessary dependencies.", 2)

        # train the model
        self.handle_log_event("Training the model.", 3)
        cmd = f"python {self.version_folder}/train.py"
        if project_name:
            cmd += f" --project {project_name}"
        if run_name:
            cmd += f" --name {run_name}"
        if start_weights_path:
            # check if the file exists
            if not Path(start_weights_path).exists():
                self.handle_log_event(f"File {start_weights_path} does not exist! It should be the path to a .pt file", 0)
            cmd += f' --weights "{start_weights_path}"'
        else:
            self.handle_log_event("No start weights provided!",0)
        if data_yaml_path:
            # check if the file exists
            if not Path(data_yaml_path).exists():
                self.handle_log_event(f"File {data_yaml_path} does not exist! It should be the path to a .yaml file", 0)
            
            # open the file and check if it has the necessary fields (train, val, test) and check if those filds hold a existent path
            with open(data_yaml_path, 'r') as stream:
                data = yaml.safe_load(stream)
                if not data:
                    self.handle_log_event(f"File {data_yaml_path} is empty!", 0)
                if 'train' not in data:
                    self.handle_log_event(f"Field 'train' not found in the file {data_yaml_path}!", 0)
                if 'val' not in data:
                #     self.handle_log_event(f"Field 'val' not found in the file {data_yaml_path}!", 0)
                # if 'test' not in data:
                    self.handle_log_event(f"Field 'test' not found in the file {data_yaml_path}!", 0)
                if not Path(data['train']).exists():
                    self.handle_log_event(f"Path {data['train']} does not exist!", 0)
                if not Path(data['val']).exists():
                    self.handle_log_event(f"Path {data['val']} does not exist!", 0)
                # if not Path(data['test']).exists():
                #     self.handle_log_event(f"Path {data['test']} does not exist!", 0)

            cmd += f' --data "{data_yaml_path}"'
        else:
            self.handle_log_event("No data yaml path provided!",0)
        if batch_size:
            cmd += f" --batch-size {batch_size}"
        if num_epochs:
            cmd += f" --epochs {num_epochs}"

        if torch.cuda.is_available():
            # if cuda is available, train with the gpu
            self.handle_log_event("cuda is available", 2)
            cmd += " --device 0"

            self.handle_log_event(f"Command to train the model: {cmd}", 3)
            if(os.system(cmd) != 0):
                self.handle_log_event(f"Error while training the model", 0)
        else:
            # if cuda is not available, train with the cpu
            if(self.force_gpu):
                self.handle_log_event("cuda is not available and `force_gpu` is present. Will not use CPU to train! Please check if your CUDA version is 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive", 0)
            self.handle_log_event("cuda is not available", 1)
            self.handle_log_event(f"Command to train the model: {cmd}", 3)
            if(os.system(cmd) != 0):
                self.handle_log_event(f"Error while training the model", 0)

        return  
    
    # def _setup(self) -> None:
    #     """Setup the YOLOv7 repository."""
    #     # install the dependencies on the requirements.txt inside the repo downloaded
    #     self.handle_log_event("Installing the dependencies of the YOLOv7 repository.", 2)
    #     os.system(f"pip install -r {self.version_folder}/requirements.txt")
    #     self.handle_log_event("Dependencies installed.", 3)

        
