from dataclasses import dataclass
import time
import os


class FileError(ValueError):
    pass


@dataclass
class Files:
    input_folder: str
    output_base_folder: str = './data/output'

    def __post_init__(self):
        # Input
        if not os.path.exists(self.input_folder):
            raise FileError("Invalid Input Data."
                            "File Path Does Not Exist")
        # TODO: Check that is a valid input with all txt needed!

        # Output
        self.output_name = time.strftime("%Y_%m_%d_%H_%M_%S")
        self.output_folder = os.path.join(
            self.output_base_folder,
            self.output_name)
        os.makedirs(self.output_folder)
