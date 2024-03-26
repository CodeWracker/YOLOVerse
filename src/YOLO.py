# THIS FILE IS REPONSABLE FOR THE FIRST LAYER OF THE YOLO API. HERE THE PADRONIZATION TAKE EFEECT


VERSIONS_REPOS = {"yolov7": "https://github.com/WongKinYiu/yolov7"}

import sys
import os
from pathlib import Path
# abstract base class work
from abc import ABC, abstractmethod


from logger import Logger

# add to path the location of this file
YOLO_FILE = Path(__file__).resolve()
ROOT_YOLO_FILE = YOLO_FILE.parents[0]
if str(ROOT_YOLO_FILE) not in sys.path:
    sys.path.append(str(ROOT_YOLO_FILE))


# check OS
if os.name == 'nt':
    OPERATING_SYSTEM = 'windows'
elif os.name == 'posix':
    OPERATING_SYSTEM = 'linux'
else:
    OPERATING_SYSTEM = 'unknown'



class BoundingBoxDetection:
    """Represents the detection of a bounding box in an image."""

    def __init__(self, x1, y1, x2, y2, conf, label_class):
        self._validate_coordinates(x1, y1, x2, y2)
        self._validate_confidence(conf)

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.conf = conf
        self.label_class = label_class
    
    def __str__(self):
        return f'{self.label_class} {self.x1} {self.y1} {self.x2}, {self.y2}'
    
    def _validate_coordinates(self, x1, y1, x2, y2):
        """Validation of the coordinates."""
        if x2 <= x1 or y2 <= y1:
            raise ValueError("INVALID COORDINATES: x2 <= x1 or y2 <= y1.")
    
    def _validate_confidence(self, conf):
        """Validation of the confidence."""
        if conf < 0 or conf > 1:
            raise ValueError("INVALID CONFIDENCE: 0 <= conf <= 1.")

    def center(self):
        """Returns the center of the detection as a tuple (x, y)."""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
    
    def area(self):
        """Returns the area of the detection as a float."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def intersection_area(self, other):
        """Returns the area of the intersection with another detection."""
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)       
        return max(0, x2 - x1) * max(0, y2 - y1)

    
    def iou(self, other):
        """Returns the Intersection over Union (IoU) of two detections."""
        intersection = self.intersection_area(other)
        union = self.area() + other.area() - intersection
        
        return 0 if union == 0 else intersection / union

    def get_movement_vector(self, future):
        """Returns the movement vector from the current detection to a future detection as a tuple (dx, dy)."""
        current_center = self.center()
        future_center = future.center()
        # x, y
        return future_center[0] - current_center[0], future_center[1] - current_center[1]
    

# Father class of the YOLO API with the main methods (Train, Detect, Resume)
class YOLO(Logger):
    """Represents the YOLO API."""

    # ATRIBUTES
    yolo_repo_download_path: str


    # -------------
    # PRIVATE METHODS
    def _validate_yolo_download_path(self, path: str) -> bool:
        """Validates the path."""
        if path is None or path == "":
            self.handle_log_event("The path parameter to download the YOLO repository is empty.", 0)
            return False
        if not Path(path).exists():
            self.handle_log_event("The path to download the YOLO repository does not exist. It will be created.", 2)
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except:
                self.handle_log_event("The path to download the YOLO repository could not be created.", 0)
                return False
        return True

    def _clone_repo(self, repo_url: str, version_folder:str) -> None:
        """Clones the YOLO repository from GitHub.

        Parameters:
        - yolo_version (str): Version of the YOLO repository to clone.
        """
        # if the provided path is not empty, download the repo else continue (it assumes that the repo is already downloaded) and log the event
        if len(os.listdir(self.yolo_repo_download_path)) == 0:
            os.system(f"git clone {repo_url} {version_folder}")
        else:
            self.handle_log_event("The path to download the YOLO repository is not empty. Assuming that it has already been downloaded to the latest version supported.", 1)

    def _download_repo(self, yolo_version: str) -> None:
        """Downloads the YOLO repository from GitHub.

        Parameters:
        - yolo_version (str): Version of the YOLO repository to download.
        """
        
        # if the versions is not a key on the versions ob
        # ject list, raise an error
        if(yolo_version not in VERSIONS_REPOS.keys()):
            self.handle_log_event(f"The provided YOLO version '{yolo_version}' is not available.", 0)

        
        # clone the folder in the download path 
        repo_url = VERSIONS_REPOS[yolo_version]
        # create a folder with the name of the version
        version_folder = Path(self.yolo_repo_download_path, yolo_version)
        # clone the repo
        self._clone_repo(repo_url, version_folder)
        # # delete the .git folder inside the folder
        self._delete_git_folder(version_folder)
    
    def _delete_git_folder(self, yolo_version_folder: str) -> None:
        """Deletes the .git folder inside the YOLO repository folder.

        Parameters:
        - yolo_version_folder (str): Path to the YOLO repository folder.
        """
        git_folder = Path(yolo_version_folder, ".git")
        if git_folder.exists():
            if(OPERATING_SYSTEM == 'windows'):
                os.system(f"rmdir /s /q {git_folder}")
            elif(OPERATING_SYSTEM == 'linux'):
                os.system(f"rm -rf {git_folder}")
            else:
                self.handle_log_event("The operating system is not supported.", 0)

        


    # -------------
    # PUBLIC METHODS

    # initialize the model
    def __init__(self, yolo_repo_download_path: str, verbosity: int = 0) -> None:
        """
        THIS CLASS ONLY CREATES AN UNIFORM API FOR EVERY YOLO IMPLEMENTATION, NOT IMPLEMENTING THE YOLO ITSELF.
        SO, IN ORDER TO USE THIS CLASS, IT NEEDS THE REFERENCE FOR THE ORIGINAL IMPLEMENTATION OF THE YOLO VERSION YOU WANT TO USE.
        THE CORRESPONDENT YOLO WILL BE DOWNLOADED TO THE PATH IN yolo_repo_download_path/{yolo_version} AND WILL BE IMPORTED IN THE CHILDREN WRAPPER CLASS
        """
        super().__init__(verbosity = verbosity)
        self.yolo_repo_download_path = yolo_repo_download_path
        self._validate_yolo_download_path(yolo_repo_download_path)


    
    def train(self, project_name: str, run_name: str, start_weights_path: str, data_yaml_path: str, batch_size: int, num_epochs: int) -> None:
        """Trains the YOLO model.

        Parameters:
        - project_name (str): Name of the project. Models will be saved to 'project_name/run_name'.
        - run_name (str): Name of the run. Models will be saved to 'project_name/run_name'.
        - start_weights_path (str): Path to the initial weights file (.pt).
        - data_yaml_path (str): Path to the data.yaml file.
        - batch_size (int): Batch size used during training.
        - num_epochs (int): Number of epochs for training.
        """
        pass

    

