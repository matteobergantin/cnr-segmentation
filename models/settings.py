from pydantic import BaseModel, model_validator, computed_field, ConfigDict
import typing
import json
import os

class ProjectSettings(BaseModel):
    RUN_MODE: typing.Literal["train", "test"] = "test"
    EVALUATE_MODEL: bool = True
    DEVICE: typing.Literal["cuda", "cpu"] = 'cpu'
    DATASET_PATH: str = "dataset"
    OUTPUT_DIR: str = "output"
    
    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def MASKS_DIR_NAME(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "masks")

    DEEPLAB_MODEL_NAME: str = "deeplabv3_resnet50"
    DEEPLAB_PRETRAINED_WEIGHTS: str = "DEFAULT"
    TRAINED_MODEL_FILENAME: str = "deeplabv3_adjusted_v2_train_weights.pth"

    @property
    def TRAINED_MODEL_PATH(self) -> str:
        return os.path.join(self.OUTPUT_DIR, 'trained_models', self.TRAINED_MODEL_FILENAME)

    BATCH_SIZE: int = 2
    EPOCHS: int = 30
    LEARNING_RATE: float = 1e-4
    TRAIN_SPLIT_RATIO: float = 0.8

    NUM_CLASSES: int = 5
    CLASS_WEIGHTS: list[float] = [0.1, 1.5, 1.25, 1.5, 3.0]

    @model_validator(mode='after')
    def _validate_model(self):
        if len(self.CLASS_WEIGHTS) != self.NUM_CLASSES:
            raise ValueError("The number of classes must be equal to the length of the classes' weights")
        
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MASKS_DIR_NAME, exist_ok=True)
        return self

    IMAGE_SIZE: int = 512

    @property
    def RESULTS_ROOT_PATH(self) -> str:
        return os.path.join(self.OUTPUT_DIR, 'results')

    @property
    def TRAIN_RESULTS_DIR(self) -> str:
        return os.path.join(self.RESULTS_ROOT_PATH, "training")

    @property
    def VAL_RESULTS_DIR(self) -> str:
        return os.path.join(self.RESULTS_ROOT_PATH, "validation")

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            raw_settings = json.load(f)
        return cls(**raw_settings)
    
    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as f:
            f.write(self.model_dump_json(indent=2))