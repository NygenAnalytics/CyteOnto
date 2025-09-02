from pathlib import Path
from unittest.mock import patch

from cyteonto.path_config import PathConfig
from cyteonto.storage.file_utils import FileManager


class TestFileManager:
    """Test FileManager functionality."""

    def test_init_default_paths(self):
        """Test FileManager initialization with default paths."""
        manager = FileManager()
        assert isinstance(manager.path_config, PathConfig)
        assert manager.path_config.base_data_path.exists()

    def test_init_custom_paths(self, temp_dir):
        """Test FileManager initialization with custom paths."""
        base_path = str(temp_dir / "base")
        user_path = str(temp_dir / "user")

        manager = FileManager(base_path, user_path)
        assert manager.path_config.base_data_path == Path(base_path)
        assert manager.path_config.user_data_path == Path(user_path)

    def test_get_embedding_file_path(self, temp_dir):
        """Test getting embedding file path."""
        manager = FileManager(str(temp_dir))

        path = manager.get_embedding_file_path(
            "test-text-model", "test-embedding-model"
        )

        assert path.parent.exists()
        assert "embeddings_test-text-model_test-embedding-model.npz" in str(path)
        assert "embedding/cell_ontology" in str(path)

    def test_get_descriptions_file_path(self, temp_dir):
        """Test getting descriptions file path."""
        manager = FileManager(str(temp_dir))

        path = manager.get_descriptions_file_path("test-text-model")

        assert path.parent.exists()
        assert "descriptions_test-text-model.json" in str(path)
        assert "embedding/descriptions" in str(path)

    def test_get_user_embeddings_path_default(self, temp_dir):
        """Test getting user embeddings path with default parameters."""
        manager = FileManager(str(temp_dir), str(temp_dir / "user"))

        path = manager.get_user_embeddings_path(
            "test-algorithm", "test-text-model", "test-embedding-model"
        )

        assert path.parent.exists()
        assert (
            "test-algorithm_embeddings_test-text-model_test-embedding-model.npz"
            in str(path)
        )
        assert "embeddings/general" in str(path)

    def test_get_user_embeddings_path_with_study(self, temp_dir):
        """Test getting user embeddings path with study name."""
        manager = FileManager(str(temp_dir), str(temp_dir / "user"))

        path = manager.get_user_embeddings_path(
            "CellTypist",
            "test-text-model",
            "test-embedding-model",
            "algorithms",
            "study1",
        )

        assert path.parent.exists()
        assert "CellTypist_embeddings_test-text-model_test-embedding-model.npz" in str(
            path
        )
        assert "embeddings/study1/algorithms" in str(path)

    def test_get_user_descriptions_path_default(self, temp_dir):
        """Test getting user descriptions path with default parameters."""
        manager = FileManager(str(temp_dir), str(temp_dir / "user"))

        path = manager.get_user_descriptions_path("test-algorithm", "test-text-model")

        assert path.parent.exists()
        assert "test-algorithm_descriptions_test-text-model.json" in str(path)
        assert "descriptions/general" in str(path)

    def test_get_user_descriptions_path_with_study(self, temp_dir):
        """Test getting user descriptions path with study name."""
        manager = FileManager(str(temp_dir), str(temp_dir / "user"))

        path = manager.get_user_descriptions_path(
            "author", "test-text-model", "author", "study1"
        )

        assert path.parent.exists()
        assert "author_descriptions_test-text-model.json" in str(path)
        assert "descriptions/study1/author" in str(path)

    def test_check_file_exists_true(self, temp_dir):
        """Test checking file existence when file exists."""
        manager = FileManager()
        test_file = temp_dir / "test.txt"
        test_file.touch()

        result = manager.check_file_exists(test_file)
        assert result is True

    def test_check_file_exists_false(self, temp_dir):
        """Test checking file existence when file doesn't exist."""
        manager = FileManager()
        test_file = temp_dir / "non_existent.txt"

        result = manager.check_file_exists(test_file)
        assert result is False

    def test_check_file_exists_directory(self, temp_dir):
        """Test checking file existence on directory."""
        manager = FileManager()
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()

        result = manager.check_file_exists(test_dir)
        assert result is False  # Directory should not be considered a file

    def test_ensure_directory_exists_new(self, temp_dir):
        """Test ensuring directory exists for new path."""
        manager = FileManager()
        nested_path = temp_dir / "deep" / "nested" / "path" / "file.txt"

        manager.ensure_directory_exists(nested_path)

        assert nested_path.parent.exists()
        assert nested_path.parent.is_dir()

    def test_ensure_directory_exists_existing(self, temp_dir):
        """Test ensuring directory exists for existing path."""
        manager = FileManager()
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir(parents=True)

        file_path = existing_dir / "file.txt"
        manager.ensure_directory_exists(file_path)

        assert existing_dir.exists()

    def test_get_ontology_mapping_path(self, temp_dir):
        """Test getting ontology mapping path."""
        manager = FileManager(str(temp_dir))

        path = manager.get_ontology_mapping_path()

        expected_path = temp_dir / "cell_ontology" / "cell_to_cell_ontology.csv"
        assert path == expected_path

    def test_get_ontology_owl_path(self, temp_dir):
        """Test getting ontology OWL path."""
        manager = FileManager(str(temp_dir))

        path = manager.get_ontology_owl_path()

        expected_path = temp_dir / "cell_ontology" / "cl.owl"
        assert path == expected_path

    @patch.object(PathConfig, "validate_core_files")
    def test_validate_data_files(self, mock_validate, temp_dir):
        """Test validating data files."""
        mock_validate.return_value = {"ontology_mapping": True, "ontology_owl": False}

        manager = FileManager(str(temp_dir))
        result = manager.validate_data_files()

        assert result == {"ontology_mapping": True, "ontology_owl": False}
        mock_validate.assert_called_once()

    def test_validate_data_files_integration(self, temp_dir):
        """Test validating data files with real files."""
        # Create ontology directory and files
        ontology_dir = temp_dir / "cell_ontology"
        ontology_dir.mkdir(parents=True)

        csv_file = ontology_dir / "cell_to_cell_ontology.csv"
        csv_file.touch()

        # Don't create OWL file

        manager = FileManager(str(temp_dir))
        result = manager.validate_data_files()

        assert result["ontology_mapping"] is True
        assert result["ontology_owl"] is False

    def test_path_config_delegation(self, temp_dir):
        """Test that FileManager properly delegates to PathConfig."""
        manager = FileManager(str(temp_dir))

        # Test that all path methods work
        embedding_path = manager.get_embedding_file_path("model1", "model2")
        descriptions_path = manager.get_descriptions_file_path("model1")
        user_emb_path = manager.get_user_embeddings_path("algo", "model1", "model2")
        user_desc_path = manager.get_user_descriptions_path("algo", "model1")
        mapping_path = manager.get_ontology_mapping_path()
        owl_path = manager.get_ontology_owl_path()

        # All should return Path objects
        assert isinstance(embedding_path, Path)
        assert isinstance(descriptions_path, Path)
        assert isinstance(user_emb_path, Path)
        assert isinstance(user_desc_path, Path)
        assert isinstance(mapping_path, Path)
        assert isinstance(owl_path, Path)

    def test_file_operations_with_special_characters(self, temp_dir):
        """Test file operations with special characters in model names."""
        manager = FileManager(str(temp_dir), str(temp_dir / "user"))

        # Model names with special characters
        text_model = "provider/model:v1.0"
        embedding_model = "embed.provider/model-name:latest"
        identifier = "algorithm.v2"

        # Should handle special characters in paths
        embedding_path = manager.get_embedding_file_path(text_model, embedding_model)
        user_path = manager.get_user_embeddings_path(
            identifier, text_model, embedding_model
        )

        assert embedding_path.parent.exists()
        assert user_path.parent.exists()

        # Check that special characters are cleaned
        assert "/" not in embedding_path.name
        assert ":" not in embedding_path.name
        assert "/" not in user_path.name
        assert ":" not in user_path.name
        assert "." in user_path.name  # Dots might be preserved in some contexts
