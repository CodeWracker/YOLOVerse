# THIS FILE IS TO ENCAPSULATE THE USAGE OF THE YOLOv6 API CREATED BY @WonkKinYiu (https://github.com/ultralytics/yolov5), forked by https://github.com/CodeWracker/yolov5

import sys
from pathlib import Path
import os
import torch
import yaml



from src.yolo import YOLO, YOLOOptions, temporary_sys_path_addition

class YOLOv5(YOLO):
    """This class is a wrapper over the YOLOv5 Implementation, standardizing the API."""

    def __init__(self, yolo_repo_download_path: str = None, verbosity:int = 0) -> None:
        super().__init__(yolo_repo_download_path=yolo_repo_download_path, verbosity=verbosity)

        # donwload the repo git git clone and delete the .git folder
        self._download_repo('yolov5')

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
        img_size: list = [640, 640],
        no_save: bool = False,
        no_test: bool = False,
        exist_ok: bool = False
    ) -> None:

        options = YOLOOptions()

        # Check if the parameters are valid
        if not self._validate_start_weights_path(weights):
            self.handle_log_event(f"A problem occurred while validating your start weights path: {weights}", 0)
        options.weights = weights

        if not self._validate_yaml_file(data):
            self.handle_log_event(f"A problem occurred while validating your YAML file path: {data}", 0)
        options.data = data

        options.device = 'cpu' if device != 0 else str(device)  # Adapting for potential non-zero GPU device IDs

        options.cfg = cfg
        options.hyp = hyp if hyp != "" else f"{self.version_folder}/data/hyps/hyp.scratch-low.yaml"
        options.batch_size = batch_size
        options.total_batch_size = batch_size
        options.epochs = num_epochs
        options.project = project_name
        options.name = run_name
        options.imgsz = img_size[0]  # Adapting for the single img_size parameter in YOLOv5
        options.nosave = no_save
        options.noval = no_test  # Adapting for the discrepancy between no_test in function and noval in argparse
        options.exist_ok = exist_ok

        # Hard coded options based on argparse defaults
        options.noplots = False
        # options.evolve_population = ROOT / "data/hyps"  # argparse uses --evolve_population with a default of ROOT / "data/hyps"
        options.resume_evolve = None  # argparse allows for an optional const=None, handling as None by default
        options.resume = False  # argparse allows for an optional const=True, handling as False by default
        options.noautoanchor = False
        options.evolve = False
        # options. = ""
        options.cache = "ram"
        options.patience = 100  # argparse uses --patience with a default of 100
        options.image_weights = False
        options.multi_scale = False
        options.single_cls = False
        options.optimizer = "SGD"  # argparse uses --optimizer with a default of 'SGD', handling Adam as False by default
        options.sync_bn = False
        options.entity = None
        options.quad = False
        options.cos_lr = False  # argparse uses --cos-lr for cosine LR scheduler, linear_lr is not directly mentioned
        options.label_smoothing = 0.0
        options.upload_dataset = False
        options.bbox_interval = -1
        options.save_period = -1
        options.artifact_alias = "latest"
        options.rect = False
        options.workers = 8
        options.freeze = [0]  # argparse default for --freeze
        options.seed = 0
        options.local_rank = -1
        options.ndjson_console = False
        options.ndjson_file = False

        # Adjusting device handling based on the argparse's device option
        if device == 'cpu' or not torch.cuda.is_available():
            self.handle_log_event("CUDA is not available, using CPU.", 0)
        else:
            self.handle_log_event("CUDA is available.", 2)


        

        with temporary_sys_path_addition(self.version_folder):
            from yolo_repo.yolov5.train import train as yolov5_train
            from yolo_repo.yolov5.utils.torch_utils import select_device
            from yolo_repo.yolov5.utils.callbacks import Callbacks
            from yolo_repo.yolov5.utils.general import increment_path
            device = select_device(options.device, batch_size=options.batch_size)
            options.save_dir = str(increment_path(Path(options.project) / options.name, exist_ok=options.exist_ok))
            callbacks = Callbacks()
            self.handle_log_event(f"Loading the hyperparameters from: {options.hyp}", 3)

            

            
            yolov5_train(options.hyp, options, device, callbacks)

        return