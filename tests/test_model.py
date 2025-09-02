import pytest
from pydantic import ValidationError

from cyteonto.model import CellDescription, to_sentence


class TestCellDescription:
    """Test CellDescription model."""

    def test_cell_description_creation(self):
        """Test creating a CellDescription instance."""
        desc = CellDescription(
            initialLabel="T cell",
            descriptiveName="Helper T cell",
            function="Immune response coordination",
            markerGenes=["CD4", "CD3"],
            diseaseRelevance="Autoimmune diseases",
            developmentalStage="Mature",
        )

        assert desc.initialLabel == "T cell"
        assert desc.descriptiveName == "Helper T cell"
        assert desc.function == "Immune response coordination"
        assert desc.markerGenes == ["CD4", "CD3"]
        assert desc.diseaseRelevance == "Autoimmune diseases"
        assert desc.developmentalStage == "Mature"

    def test_cell_description_validation(self):
        """Test CellDescription field validation."""
        # Test with missing required fields
        with pytest.raises(ValidationError):
            CellDescription()

        # Test with invalid types
        with pytest.raises(ValidationError):
            CellDescription(
                initialLabel="T cell",
                descriptiveName="Helper T cell",
                function="Immune response",
                markerGenes="CD4",  # Should be a list
                diseaseRelevance="Autoimmune",
                developmentalStage="Mature",
            )

    def test_get_blank(self):
        """Test get_blank class method."""
        blank = CellDescription.get_blank()

        assert blank.initialLabel == "Monocytes"
        assert blank.descriptiveName == ""
        assert blank.function == ""
        assert blank.markerGenes == []
        assert blank.diseaseRelevance == ""
        assert blank.developmentalStage == ""

    def test_get_example(self):
        """Test get_example class method."""
        example = CellDescription.get_example()
        # Currently returns blank, but structure is tested
        assert isinstance(example, CellDescription)
        assert example.initialLabel == "Monocytes"

    def test_to_sentence(self):
        """Test to_sentence function."""
        desc = CellDescription(
            initialLabel="T cell",
            descriptiveName="CD4+ helper T cell",
            function="Coordinates immune responses",
            markerGenes=["CD4", "CD3", "TCR"],
            diseaseRelevance="Critical in autoimmune diseases",
            developmentalStage="Mature adaptive immune cell",
        )

        sentence = to_sentence(desc)

        assert "T cell is CD4+ helper T cell" in sentence
        assert "Coordinates immune responses" in sentence
        assert "Critical in autoimmune diseases" in sentence
        assert "Mature adaptive immune cell" in sentence
        assert "CD4, CD3, TCR" in sentence

    def test_to_sentence_empty_markers(self):
        """Test to_sentence with empty marker genes."""
        desc = CellDescription(
            initialLabel="Unknown cell",
            descriptiveName="Uncharacterized cell",
            function="Unknown function",
            markerGenes=[],
            diseaseRelevance="No known relevance",
            developmentalStage="Unknown stage",
        )

        sentence = to_sentence(desc)

        assert "Unknown cell is Uncharacterized cell" in sentence
        assert "marker genes are " in sentence  # Empty list should be handled

    def test_model_dump_and_load(self, sample_cell_description):
        """Test serialization and deserialization."""
        # Test model_dump
        data = sample_cell_description.model_dump()

        assert isinstance(data, dict)
        assert data["initialLabel"] == "T cell"
        assert data["markerGenes"] == ["CD4", "CD3", "TCR"]

        # Test model_validate
        reconstructed = CellDescription.model_validate(data)
        assert reconstructed == sample_cell_description

    def test_equality(self):
        """Test CellDescription equality."""
        desc1 = CellDescription(
            initialLabel="T cell",
            descriptiveName="Helper T cell",
            function="Immune response",
            markerGenes=["CD4"],
            diseaseRelevance="Autoimmune",
            developmentalStage="Mature",
        )

        desc2 = CellDescription(
            initialLabel="T cell",
            descriptiveName="Helper T cell",
            function="Immune response",
            markerGenes=["CD4"],
            diseaseRelevance="Autoimmune",
            developmentalStage="Mature",
        )

        desc3 = CellDescription(
            initialLabel="B cell",
            descriptiveName="Memory B cell",
            function="Antibody production",
            markerGenes=["CD19"],
            diseaseRelevance="Immunodeficiency",
            developmentalStage="Mature",
        )

        assert desc1 == desc2
        assert desc1 != desc3
