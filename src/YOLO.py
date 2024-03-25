# THIS FILE IS REPONSABLE FOR THE FIRST LAYER OF THE YOLO API. HERE THE PADRONIZATION TAKE EFEECT


# abstract base class work
from abc import ABC, abstractmethod

class BoundingBoxDetection:
    """Represents the detection of a bounding box in an image."""

    def __init__(self, x1, y1, x2, y2, conf, label_class):
        self._validate_coordinates(x1, y1, x2, y2)
        self._validate_confidence(conf)

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.conf = conf
        self.label_class = label_class
    
    def __str__(self):
        return f'[{self.label_class}]: ({self.x1}, {self.y1}) ({self.x2}, {self.y2})'
    
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
    

# Father class of the YOLO API with the main methods (Train, Predict, Resume)
class YOLO(ABC):
    """Represents the YOLO API."""

    @abstractmethod
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

    



