from dataclasses import dataclass
import time
import os


class FileError(ValueError):
    pass


@dataclass
class Files:
    datasets_folder: str
    output_base_folder: str = './data/output'
    models_output_path: str = './data/models'

    def __post_init__(self):
        # Input
        if not os.path.exists(self.datasets_folder):
            os.makedirs(self.datasets_folder)
        self.celeba_dataset_filepath = os.path.join(
            self.datasets_folder,
            'celeba')
        if not os.path.exists(self.celeba_dataset_filepath):
            raise FileError("Invalid Input Data."
                            "Dataset CelebA not found.")
        # Output
        self.output_name = time.strftime("%Y_%m_%d_%H_%M_%S")
        self.output_folder = os.path.join(
            self.output_base_folder,
            self.output_name)
        os.makedirs(self.output_folder)
        # Models
        if not os.path.exists(self.models_output_path):
            os.makedirs(self.models_output_path)
