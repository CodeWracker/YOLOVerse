from src.yolov7 import YOLOv7
from src.logger import Logger

# my_model = YOLOv7()
# my_model.train(
#     project_name='project_name',
#     run_name='run_name',
#     start_weights_path='start_weights_path',
#     data_yaml_path='data_yaml_path',
#     batch_size=8,
#     num_epochs=100
# )

logger = Logger(verbosity=3)
logger.handle_event('An event happened', 3)
logger.handle_event('Another event happened', 2)
logger.handle_event('Yet another event happened', 1)
logger.handle_event('One more event happened', 0)