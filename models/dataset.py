from ptdataset import LABEL_NORMALIZATION_DICT
import pydantic
import typing

class DatasetShapeJSON(pydantic.BaseModel):
  label: typing.Literal['vegetazione', 'muratura_mancante', 'nessun_difetto', 'fratturazione_fessurazione']
  points: list[tuple[float,float]]
  group_id: int | None
  shape_type: str
  flags: dict

  @pydantic.field_validator("label", mode='before')
  @classmethod
  def _normalize_label(cls, v):
    return LABEL_NORMALIZATION_DICT[v]

class DatasetLabelMeModel(pydantic.BaseModel):
  version: str
  flags: dict
  shapes: list[DatasetShapeJSON]
  imagePath: str
  # Ignore imageData, too heavy
  #imageData: str
  imageHeight: int
  imageWidth: int