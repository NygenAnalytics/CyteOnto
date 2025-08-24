# cyteonto/model.py

from pydantic import BaseModel, Field


class CellDescription(BaseModel):
    initialLabel: str = Field(description="The initial label for the cell")
    descriptiveName: str = Field(description="A descriptive name for the cell")
    function: str = Field(description="The function of the cell")
    markerGenes: list[str] = Field(description="Marker genes associated with the cell")
    diseaseRelevance: str = Field(description="Disease relevance of the cell")
    developmentalStage: str = Field(description="Developmental stage of the cell")

    @classmethod
    def get_blank(cls) -> "CellDescription":
        return cls(
            initialLabel="Monocytes",
            descriptiveName="",
            function="",
            markerGenes=[],
            diseaseRelevance="",
            developmentalStage="",
        )

    @classmethod
    def get_example(cls) -> "CellDescription":
        # PLACEHOLDER
        return cls.get_blank()

    @classmethod
    def to_sentence(cls) -> str:
        return (
            f"{cls.initialLabel} is {cls.descriptiveName}. {cls.function}. "
            f"{cls.diseaseRelevance}. {cls.developmentalStage}. "
            f"The marker genes are {', '.join(cls.markerGenes)}."
        )
