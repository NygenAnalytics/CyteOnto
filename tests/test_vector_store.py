import json
from unittest.mock import patch

import numpy as np

from cyteonto.model import CellDescription
from cyteonto.storage.vector_store import VectorStore


class TestVectorStore:
    """Test VectorStore functionality."""

    def test_init(self):
        """Test VectorStore initialization."""
        store = VectorStore()
        assert store is not None

    def test_save_embeddings_success(
        self, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test successful embeddings saving."""
        store = VectorStore()
        filepath = temp_dir / "test_embeddings.npz"

        result = store.save_embeddings(sample_embeddings, sample_ontology_ids, filepath)

        assert result is True
        assert filepath.exists()

        # Verify file content
        data = np.load(filepath, allow_pickle=True)
        assert "embeddings" in data
        assert "ontology_ids" in data
        assert "metadata" in data

        np.testing.assert_array_equal(data["embeddings"], sample_embeddings)
        assert list(data["ontology_ids"]) == sample_ontology_ids
        assert data["metadata"].item()["num_embeddings"] == len(sample_embeddings)

    def test_save_embeddings_creates_directory(
        self, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test that save_embeddings creates parent directories."""
        store = VectorStore()
        nested_dir = temp_dir / "nested" / "deep" / "path"
        filepath = nested_dir / "test_embeddings.npz"

        result = store.save_embeddings(sample_embeddings, sample_ontology_ids, filepath)

        assert result is True
        assert filepath.exists()
        assert nested_dir.exists()

    def test_save_embeddings_empty_array(self, temp_dir, sample_ontology_ids):
        """Test saving empty embeddings array."""
        store = VectorStore()
        empty_embeddings = np.array([]).reshape(0, 768)
        filepath = temp_dir / "empty_embeddings.npz"

        result = store.save_embeddings(empty_embeddings, [], filepath)

        assert result is True
        assert filepath.exists()

        # Verify metadata for empty array
        data = np.load(filepath, allow_pickle=True)
        metadata = data["metadata"].item()
        assert metadata["num_embeddings"] == 0
        assert metadata["embedding_dim"] == 0

    @patch("numpy.savez_compressed")
    def test_save_embeddings_failure(
        self, mock_savez, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test embeddings saving failure."""
        mock_savez.side_effect = Exception("Save failed")
        store = VectorStore()
        filepath = temp_dir / "test_embeddings.npz"

        result = store.save_embeddings(sample_embeddings, sample_ontology_ids, filepath)
        assert result is False

    def test_load_embeddings_success(
        self, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test successful embeddings loading."""
        store = VectorStore()
        filepath = temp_dir / "test_embeddings.npz"

        # First save embeddings
        store.save_embeddings(sample_embeddings, sample_ontology_ids, filepath)

        # Then load them
        result = store.load_embeddings(filepath)

        assert result is not None
        embeddings, ontology_ids, metadata = result

        np.testing.assert_array_equal(embeddings, sample_embeddings)
        assert ontology_ids == sample_ontology_ids
        assert isinstance(metadata, dict)
        assert metadata["num_embeddings"] == len(sample_embeddings)

    def test_load_embeddings_file_not_found(self, temp_dir):
        """Test loading embeddings when file doesn't exist."""
        store = VectorStore()
        filepath = temp_dir / "non_existent.npz"

        result = store.load_embeddings(filepath)
        assert result is None

    @patch("numpy.load")
    def test_load_embeddings_failure(self, mock_load, temp_dir):
        """Test embeddings loading failure."""
        mock_load.side_effect = Exception("Load failed")
        store = VectorStore()
        filepath = temp_dir / "test_embeddings.npz"
        filepath.touch()  # Create empty file

        result = store.load_embeddings(filepath)
        assert result is None

    def test_save_descriptions_success(self, temp_dir, sample_descriptions):
        """Test successful descriptions saving."""
        store = VectorStore()
        filepath = temp_dir / "test_descriptions.json"

        result = store.save_descriptions(sample_descriptions, filepath)

        assert result is True
        assert filepath.exists()

        # Verify file content
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == len(sample_descriptions)
        for key in sample_descriptions.keys():
            assert key in data
            assert data[key]["initialLabel"] == sample_descriptions[key].initialLabel

    def test_save_descriptions_creates_directory(self, temp_dir, sample_descriptions):
        """Test that save_descriptions creates parent directories."""
        store = VectorStore()
        nested_dir = temp_dir / "nested" / "descriptions"
        filepath = nested_dir / "test_descriptions.json"

        result = store.save_descriptions(sample_descriptions, filepath)

        assert result is True
        assert filepath.exists()
        assert nested_dir.exists()

    def test_save_descriptions_empty_dict(self, temp_dir):
        """Test saving empty descriptions dictionary."""
        store = VectorStore()
        filepath = temp_dir / "empty_descriptions.json"

        result = store.save_descriptions({}, filepath)

        assert result is True
        assert filepath.exists()

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == {}

    @patch("builtins.open", side_effect=IOError("Write failed"))
    def test_save_descriptions_failure(self, mock_open, temp_dir, sample_descriptions):
        """Test descriptions saving failure."""
        store = VectorStore()
        filepath = temp_dir / "test_descriptions.json"

        result = store.save_descriptions(sample_descriptions, filepath)
        assert result is False

    def test_load_descriptions_success(self, temp_dir, sample_descriptions):
        """Test successful descriptions loading."""
        store = VectorStore()
        filepath = temp_dir / "test_descriptions.json"

        # First save descriptions
        store.save_descriptions(sample_descriptions, filepath)

        # Then load them
        result = store.load_descriptions(filepath)

        assert result is not None
        assert len(result) == len(sample_descriptions)

        for key in sample_descriptions.keys():
            assert key in result
            assert isinstance(result[key], CellDescription)
            assert result[key].initialLabel == sample_descriptions[key].initialLabel

    def test_load_descriptions_file_not_found(self, temp_dir):
        """Test loading descriptions when file doesn't exist."""
        store = VectorStore()
        filepath = temp_dir / "non_existent.json"

        result = store.load_descriptions(filepath)
        assert result is None

    def test_load_descriptions_invalid_json(self, temp_dir):
        """Test loading descriptions with invalid JSON."""
        store = VectorStore()
        filepath = temp_dir / "invalid.json"

        # Create invalid JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("invalid json content {")

        result = store.load_descriptions(filepath)
        assert result is None

    @patch("builtins.open", side_effect=IOError("Read failed"))
    def test_load_descriptions_failure(self, mock_open, temp_dir):
        """Test descriptions loading failure."""
        store = VectorStore()
        filepath = temp_dir / "test_descriptions.json"
        filepath.touch()  # Create empty file

        result = store.load_descriptions(filepath)
        assert result is None

    def test_check_embedding_file_exists_valid(
        self, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test checking valid embedding file existence."""
        store = VectorStore()
        filepath = temp_dir / "test_embeddings.npz"

        # Save embeddings first
        store.save_embeddings(sample_embeddings, sample_ontology_ids, filepath)

        result = store.check_embedding_file_exists(filepath)
        assert result is True

    def test_check_embedding_file_exists_not_found(self, temp_dir):
        """Test checking non-existent embedding file."""
        store = VectorStore()
        filepath = temp_dir / "non_existent.npz"

        result = store.check_embedding_file_exists(filepath)
        assert result is False

    def test_check_embedding_file_exists_invalid_structure(self, temp_dir):
        """Test checking embedding file with invalid structure."""
        store = VectorStore()
        filepath = temp_dir / "invalid_embeddings.npz"

        # Create NPZ file with missing keys
        np.savez_compressed(filepath, some_data=np.array([1, 2, 3]))

        result = store.check_embedding_file_exists(filepath)
        assert result is False

    def test_check_embedding_file_exists_corrupted(self, temp_dir):
        """Test checking corrupted embedding file."""
        store = VectorStore()
        filepath = temp_dir / "corrupted.npz"

        # Create corrupted file
        with open(filepath, "w") as f:
            f.write("not a valid npz file")

        result = store.check_embedding_file_exists(filepath)
        assert result is False

    def test_round_trip_embeddings(
        self, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test round-trip save and load of embeddings."""
        store = VectorStore()
        filepath = temp_dir / "round_trip.npz"

        # Save embeddings
        save_result = store.save_embeddings(
            sample_embeddings, sample_ontology_ids, filepath
        )
        assert save_result is True

        # Load embeddings
        load_result = store.load_embeddings(filepath)
        assert load_result is not None

        loaded_embeddings, loaded_ids, loaded_metadata = load_result

        # Verify data integrity
        np.testing.assert_array_equal(loaded_embeddings, sample_embeddings)
        assert loaded_ids == sample_ontology_ids
        assert loaded_metadata["num_embeddings"] == len(sample_embeddings)
        assert loaded_metadata["embedding_dim"] == sample_embeddings.shape[1]

    def test_round_trip_descriptions(self, temp_dir, sample_descriptions):
        """Test round-trip save and load of descriptions."""
        store = VectorStore()
        filepath = temp_dir / "round_trip.json"

        # Save descriptions
        save_result = store.save_descriptions(sample_descriptions, filepath)
        assert save_result is True

        # Load descriptions
        load_result = store.load_descriptions(filepath)
        assert load_result is not None

        # Verify data integrity
        assert len(load_result) == len(sample_descriptions)
        for key, original_desc in sample_descriptions.items():
            loaded_desc = load_result[key]
            assert loaded_desc.initialLabel == original_desc.initialLabel
            assert loaded_desc.descriptiveName == original_desc.descriptiveName
            assert loaded_desc.function == original_desc.function
            assert loaded_desc.markerGenes == original_desc.markerGenes
            assert loaded_desc.diseaseRelevance == original_desc.diseaseRelevance
            assert loaded_desc.developmentalStage == original_desc.developmentalStage
