# THIS FILE IS TO ENCAPSULATE THE USAGE OF THE YOLOv7 API CREATED BY @WonkKinYiu (https://github.com/WongKinYiu/yolov7), forked by https://github.com/CodeWracker/yolov7

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
from logger import Logger


class YOLOv7Options(dict):
    """A class to represent the options of the YOLOv7 model."""

    def __init__(self) -> None:
        super().__init__()
    
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'YOLOv7Options' object has no attribute '{item}'")
            
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'YOLOv7Options' object has no attribute '{item}'")
            


class YOLOv7(YOLO):
    """This class is a wrapper over the YOLOv7 Implementation, standardizing the API."""

    def __init__(self, yolo_repo_download_path: str = None, verbosity:int = 0) -> None:
        super().__init__(yolo_repo_download_path=yolo_repo_download_path, verbosity=verbosity)


        # donwload the repo git git clone and delete the .git folder
        self._download_repo('yolov7')
        # self._setup()

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
            resume: bool = False,
            no_save: bool = False,
            no_test: bool = False,
            no_autoanchor: bool = False,
            evolve: bool = False,
            bucket: str = "",
            cache_images: bool = False,
            image_weights: bool = False,
            multi_scale: bool = False,
            single_cls: bool = False,
            adam: bool = False,
            sync_bn: bool = False,
            workers: int = 8,
            entity: str = None,
            exist_ok: bool = False,
            quad: bool = False,
            linear_lr: bool = False,
            label_smoothing: float = 0.0,
            upload_dataset: bool = False,
            bbox_interval: int = -1,
            save_period: int = -1,
            artifact_alias: str = "latest",
            freeze: list = [0],
            v5_metric: bool = False,
            rect: bool = False,

    ) -> None:
        # import the necessarie dependencies
        self.handle_log_event("Importing the necessary dependencies.", 2)

        # train the model
        self.handle_log_event("Training the model.", 3)

        options = YOLOv7Options()
        

        # check if the parameters are valid

        if not self._validate_start_weights_path(weights):
            self.handle_log_event(f"A problem ocurred while validating your start weights path: {weights}", 0)
        options.weights = weights

        if not self._validate_yaml_file(data):
            self.handle_log_event(f"A problem ocurred while validating your YAML file path: {data}", 0)
        options.data = data

        if device != 0:
            options.device = 'cpu'
        else:
            options.device = '0'

        options.cfg = cfg
        if hyp == "":
            options.hyp = f"{self.version_folder}/data/hyp.scratch.p5.yaml"
        else:
            options.hyp = hyp
        options.batch_size = batch_size
        options.total_batch_size = batch_size
        options.epochs = num_epochs
        options.project = project_name
        options.name = run_name
        options.img_size = img_size
        options.resume = resume
        options.exist_ok = exist_ok
        options.multi_scale = multi_scale
        options.single_cls = single_cls
        options.adam = adam
        options.sync_bn = sync_bn
        options.workers = workers
        options.entity = entity
        options.quad = quad
        options.linear_lr = linear_lr
        options.label_smoothing = label_smoothing
        options.upload_dataset = upload_dataset
        options.bbox_interval = bbox_interval
        options.save_period = save_period
        options.artifact_alias = artifact_alias
        options.freeze = freeze
        options.v5_metric = v5_metric
        options.nosave = no_save
        options.notest = no_test
        options.noautoanchor = no_autoanchor
        options.evolve = evolve
        options.bucket = bucket
        options.cache_images = cache_images
        options.image_weights = image_weights
        options.rect = rect
        
        options.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        options.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

        if options.device == 0:
            # if cuda is available, train with the gpu
            if torch.cuda.is_available():
                self.handle_log_event("cuda is available", 2)
            else:
                self.handle_log_event("cuda is not available", 0)

        # train the model
        # include to path the path of the repo
        sys.path.append(self.version_folder)
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
            yolov7_train(hyp_yaml, options, device, None)



        return  
    
    # def _setup(self) -> None:
    #     """Setup the YOLOv7 repository."""
    #     # install the dependencies on the requirements.txt inside the repo downloaded
    #     self.handle_log_event("Installing the dependencies of the YOLOv7 repository.", 2)
    #     os.system(f"pip install -r {self.version_folder}/requirements.txt")
    #     self.handle_log_event("Dependencies installed.", 3)

        
