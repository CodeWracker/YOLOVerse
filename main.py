from src.yolov7 import YOLOv7

my_model = YOLOv7()
my_model.train(
    project_name='project_name',
    run_name='run_name',
    start_weights_path='start_weights_path',
    data_yaml_path='data_yaml_path',
    batch_size=8,
    num_epochs=100
)