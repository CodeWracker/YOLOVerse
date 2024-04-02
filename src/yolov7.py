# THIS FILE IS TO ENCAPSULATE THE USAGE OF THE YOLOv7 API CREATED BY @WonkKinYiu (https://github.com/WongKinYiu/yolov7), forked by https://github.com/CodeWracker/yolov7

import sys
from pathlib import Path
import os
import torch
import yaml




from src.yolo import YOLO, YOLOOptions,temporary_sys_path_addition


class YOLOv7(YOLO):
    """This class is a wrapper over the YOLOv7 Implementation, standardizing the API."""

    def __init__(self, yolo_repo_download_path: str = None, verbosity:int = 0) -> None:
        super().__init__(yolo_repo_download_path=yolo_repo_download_path, verbosity=verbosity)


        # donwload the repo git git clone and delete the .git folder
        self._download_repo('yolov7')

    # override the method to train
    def train(
            self,
            project_name: str,
            run_name: str,
            weights: str,
            data: str,
            batch_size: int,
            num_epochs: int,
            device: int,
            cfg: str = "",
            hyp: str = "",
            img_size: list = [640,640],
            no_save: bool = False,
            no_test: bool = False,
            exist_ok: bool = False
    ) -> None:

        

        options = YOLOOptions()
        

        # check if the parameters are valid
        if not self._validate_start_weights_path(weights):
            self.handle_log_event(f"A problem occurred while validating your start weights path: {weights}", 0)
        options.weights = weights

        if not self._validate_yaml_file(data):
            self.handle_log_event(f"A problem occurred while validating your YAML file path: {data}", 0)
        options.data = data

        options.device = 'cpu' if device != 0 else '0'

        options.cfg = cfg
        options.hyp = hyp if hyp != "" else f"{self.version_folder}/data/hyp.scratch.p5.yaml"
        options.batch_size = batch_size
        options.total_batch_size = batch_size
        options.epochs = num_epochs
        options.project = project_name
        options.name = run_name
        options.img_size = img_size
        options.nosave = no_save
        options.notest = no_test
        options.exist_ok = exist_ok

        # Hard coded options 
        options.resume = False
        options.noautoanchor = False
        options.evolve = False
        options.bucket = ""
        options.cache_images = False
        options.image_weights = False
        options.multi_scale = False
        options.single_cls = False
        options.adam = False
        options.sync_bn = False
        options.entity = None
        options.quad = False
        options.linear_lr = False
        options.label_smoothing = 0.0
        options.upload_dataset = False
        options.bbox_interval = -1
        options.save_period = -1
        options.artifact_alias = "latest"
        options.v5_metric = False  
        options.workers = 8 
        options.freeze = [0]  
        options.rect = False

        
        options.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        options.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

        if options.device == 0:
            # if cuda is available, train with the gpu
            if torch.cuda.is_available():
                self.handle_log_event("cuda is available", 2)
            else:
                self.handle_log_event("cuda is not available", 0)

        # train the model
         

        with temporary_sys_path_addition(self.version_folder):
            self.handle_log_event(f'The path now is: {sys.path}', 1)

            from yolo_repo.yolov7.train import train as yolov7_train
            from yolo_repo.yolov7.utils.general import increment_path
            from yolo_repo.yolov7.utils.torch_utils import select_device

            self.handle_log_event(f"Loading the hyperparameters from: {options.hyp}", 3)
            with open(options.hyp) as f:
                hyp_yaml = yaml.load(f, Loader=yaml.FullLoader)
            if hyp_yaml is None:
                self.handle_log_event(f"A problem ocurred while loading the hyperparameters: {options.hyp}", 0)

            device = select_device(options.device, batch_size=options.batch_size)


            if options.resume:
                self.handle_log_event("Resuming a training is not supported yet.", 0)
            else:
                options.save_dir = increment_path(Path(options.project) / options.name, exist_ok=options.exist_ok)
                # train the model
                self.handle_log_event("Training the model.", 3)
                yolov7_train(hyp_yaml, options, device, None)


        return  
    

        
