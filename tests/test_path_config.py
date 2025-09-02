from pathlib import Path

from cyteonto.path_config import PathConfig


class TestPathConfig:
    """Test PathConfig functionality."""

    def test_init_default_paths(self):
        """Test initialization with default paths."""
        config = PathConfig()

        assert config.base_data_path.exists()
        assert config.user_data_path.exists()
        assert "data" in str(config.base_data_path)
        assert "user_files" in str(config.user_data_path)

    def test_init_custom_paths(self, temp_dir):
        """Test initialization with custom paths."""
        base_path = str(temp_dir / "custom_base")
        user_path = str(temp_dir / "custom_user")

        config = PathConfig(base_path, user_path)

        assert config.base_data_path == Path(base_path)
        assert config.user_data_path == Path(user_path)
        assert config.base_data_path.exists()
        assert config.user_data_path.exists()

    def test_init_mixed_paths(self, temp_dir):
        """Test initialization with one custom path."""
        base_path = str(temp_dir / "custom_base")

        # Only provide base path, user path should be relative to base
        config = PathConfig(base_path, None)

        assert config.base_data_path == Path(base_path)
        assert config.user_data_path == Path(base_path) / "user_files"
        assert config.base_data_path.exists()
        assert config.user_data_path.exists()

    def test_get_ontology_embedding_path(self, temp_dir):
        """Test getting ontology embedding path."""
        config = PathConfig(str(temp_dir))

        path = config.get_ontology_embedding_path(
            "test-text-model", "test-embedding-model"
        )

        assert path.parent.exists()
        assert "embeddings_test-text-model_test-embedding-model.npz" in str(path)
        assert "embedding/cell_ontology" in str(path)

    def test_get_ontology_descriptions_path(self, temp_dir):
        """Test getting ontology descriptions path."""
        config = PathConfig(str(temp_dir))

        path = config.get_ontology_descriptions_path("test-text-model")

        assert path.parent.exists()
        assert "descriptions_test-text-model.json" in str(path)
        assert "embedding/descriptions" in str(path)

    def test_get_user_embeddings_path_default(self, temp_dir):
        """Test getting user embeddings path with default category."""
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))

        path = config.get_user_embeddings_path(
            "test-algorithm", "test-text-model", "test-embedding-model"
        )

        assert path.parent.exists()
        assert (
            "test-algorithm_embeddings_test-text-model_test-embedding-model.npz"
            in str(path)
        )
        assert "embeddings/general" in str(path)

    def test_get_user_embeddings_path_with_category(self, temp_dir):
        """Test getting user embeddings path with specific category."""
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))

        path = config.get_user_embeddings_path(
            "CellTypist", "test-text-model", "test-embedding-model", "algorithms"
        )

        assert path.parent.exists()
        assert "CellTypist_embeddings_test-text-model_test-embedding-model.npz" in str(
            path
        )
        assert "embeddings/algorithms" in str(path)

    def test_get_user_embeddings_path_with_study(self, temp_dir):
        """Test getting user embeddings path with study name."""
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))

        path = config.get_user_embeddings_path(
            "author", "test-text-model", "test-embedding-model", "author", "study1"
        )

        assert path.parent.exists()
        assert "author_embeddings_test-text-model_test-embedding-model.npz" in str(path)
        assert "embeddings/study1/author" in str(path)

    def test_get_user_descriptions_path_default(self, temp_dir):
        """Test getting user descriptions path with default category."""
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))

        path = config.get_user_descriptions_path("test-algorithm", "test-text-model")

        assert path.parent.exists()
        assert "test-algorithm_descriptions_test-text-model.json" in str(path)
        assert "descriptions/general" in str(path)

    def test_get_user_descriptions_path_with_category_and_study(self, temp_dir):
        """Test getting user descriptions path with category and study."""
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))

        path = config.get_user_descriptions_path(
            "CellTypist", "test-text-model", "algorithms", "study1"
        )

        assert path.parent.exists()
        assert "CellTypist_descriptions_test-text-model.json" in str(path)
        assert "descriptions/study1/algorithms" in str(path)

    def test_get_ontology_mapping_path(self, temp_dir):
        """Test getting ontology mapping path."""
        config = PathConfig(str(temp_dir))

        path = config.get_ontology_mapping_path()

        expected_path = temp_dir / "cell_ontology" / "cell_to_cell_ontology.csv"
        assert path == expected_path

    def test_get_ontology_owl_path(self, temp_dir):
        """Test getting ontology OWL path."""
        config = PathConfig(str(temp_dir))

        path = config.get_ontology_owl_path()

        expected_path = temp_dir / "cell_ontology" / "cl.owl"
        assert path == expected_path

    def test_validate_core_files_all_exist(self, temp_dir):
        """Test validating core files when all exist."""
        config = PathConfig(str(temp_dir))

        # Create the required files
        ontology_dir = temp_dir / "cell_ontology"
        ontology_dir.mkdir(parents=True)
        (ontology_dir / "cell_to_cell_ontology.csv").touch()
        (ontology_dir / "cl.owl").touch()

        result = config.validate_core_files()

        assert result["ontology_mapping"] is True
        assert result["ontology_owl"] is True

    def test_validate_core_files_some_missing(self, temp_dir):
        """Test validating core files when some are missing."""
        config = PathConfig(str(temp_dir))

        # Create only the CSV file
        ontology_dir = temp_dir / "cell_ontology"
        ontology_dir.mkdir(parents=True)
        (ontology_dir / "cell_to_cell_ontology.csv").touch()
        # Don't create OWL file

        result = config.validate_core_files()

        assert result["ontology_mapping"] is True
        assert result["ontology_owl"] is False

    def test_validate_core_files_none_exist(self, temp_dir):
        """Test validating core files when none exist."""
        config = PathConfig(str(temp_dir))

        result = config.validate_core_files()

        assert result["ontology_mapping"] is False
        assert result["ontology_owl"] is False

    def test_clean_model_name(self, temp_dir):
        """Test model name cleaning."""
        config = PathConfig(str(temp_dir))

        # Test various special characters
        assert config._clean_model_name("provider/model") == "provider-model"
        assert config._clean_model_name("model:v1.0") == "model-v1.0"
        assert config._clean_model_name("model with spaces") == "model_with_spaces"
        assert (
            config._clean_model_name("provider/model:v1.0 beta")
            == "provider-model-v1.0_beta"
        )

    def test_clean_identifier(self, temp_dir):
        """Test identifier cleaning."""
        config = PathConfig(str(temp_dir))

        # Test various special characters
        assert config._clean_identifier("algorithm/v1") == "algorithm-v1"
        assert config._clean_identifier("algorithm:name") == "algorithm-name"
        assert (
            config._clean_identifier("algorithm with spaces") == "algorithm_with_spaces"
        )
        assert config._clean_identifier("algorithm.v2") == "algorithm_v2"
        assert (
            config._clean_identifier("complex/algorithm:v1.0 beta")
            == "complex-algorithm-v1_0_beta"
        )

    def test_path_creation_integration(self, temp_dir):
        """Test that all path creation methods work together."""
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))

        # Test all path creation methods
        embedding_path = config.get_ontology_embedding_path("model/text", "model/embed")
        desc_path = config.get_ontology_descriptions_path("model/text")
        user_emb_path = config.get_user_embeddings_path(
            "algo", "model/text", "model/embed", "algorithms", "study1"
        )
        user_desc_path = config.get_user_descriptions_path(
            "algo", "model/text", "algorithms", "study1"
        )

        # All paths should have created directories
        assert embedding_path.parent.exists()
        assert desc_path.parent.exists()
        assert user_emb_path.parent.exists()
        assert user_desc_path.parent.exists()

        # All paths should be under the correct base directories
        assert str(embedding_path).startswith(str(temp_dir))
        assert str(desc_path).startswith(str(temp_dir))
        assert str(user_emb_path).startswith(str(temp_dir / "user"))
        assert str(user_desc_path).startswith(str(temp_dir / "user"))

    def test_directory_creation_with_permissions(self, temp_dir):
        """Test directory creation with proper permissions."""
        config = PathConfig(str(temp_dir))

        # Create a deeply nested path
        path = config.get_ontology_embedding_path(
            "deeply/nested/model", "another/nested/model"
        )

        # Directory should be created with proper permissions
        assert path.parent.exists()
        assert path.parent.is_dir()

        # Test that we can write to the created directory
        test_file = path.parent / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()

    def test_path_uniqueness(self, temp_dir):
        """Test that different parameters produce different paths."""
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))

        # Different model names should produce different paths
        path1 = config.get_ontology_embedding_path("model1", "embed1")
        path2 = config.get_ontology_embedding_path("model2", "embed2")
        assert path1 != path2

        # Different identifiers should produce different paths
        user_path1 = config.get_user_embeddings_path("algo1", "model", "embed")
        user_path2 = config.get_user_embeddings_path("algo2", "model", "embed")
        assert user_path1 != user_path2

        # Different studies should produce different paths
        study_path1 = config.get_user_embeddings_path(
            "algo", "model", "embed", "general", "study1"
        )
        study_path2 = config.get_user_embeddings_path(
            "algo", "model", "embed", "general", "study2"
        )
        assert study_path1 != study_path2
