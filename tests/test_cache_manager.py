import json
from datetime import datetime
from unittest.mock import Mock

import numpy as np

from cyteonto.storage.cache_manager import CacheManager, CacheMetadata
from cyteonto.storage.vector_store import VectorStore


class TestCacheMetadata:
    """Test CacheMetadata functionality."""

    def test_cache_metadata_creation(self):
        """Test creating CacheMetadata instance."""
        creation_time = datetime.now()
        metadata = CacheMetadata(
            labels_hash="abc123",
            model_config={"model": "test", "provider": "test"},
            creation_time=creation_time,
            file_size=1024,
            version="1.0",
        )

        assert metadata.labels_hash == "abc123"
        assert metadata.model_config == {"model": "test", "provider": "test"}
        assert metadata.creation_time == creation_time
        assert metadata.file_size == 1024
        assert metadata.version == "1.0"

    def test_to_dict(self):
        """Test converting CacheMetadata to dictionary."""
        creation_time = datetime(2023, 1, 1, 12, 0, 0)
        metadata = CacheMetadata(
            labels_hash="def456",
            model_config={"embedding_model": "test-embedding"},
            creation_time=creation_time,
            file_size=2048,
        )

        result = metadata.to_dict()

        assert result["labels_hash"] == "def456"
        assert result["model_config"] == {"embedding_model": "test-embedding"}
        assert result["creation_time"] == "2023-01-01T12:00:00"
        assert result["file_size"] == 2048
        assert result["version"] == "1.0"

    def test_from_dict(self):
        """Test creating CacheMetadata from dictionary."""
        data = {
            "labels_hash": "ghi789",
            "model_config": {"text_model": "test-text"},
            "creation_time": "2023-01-01T12:00:00",
            "file_size": 512,
            "version": "1.0",
        }

        metadata = CacheMetadata.from_dict(data)

        assert metadata.labels_hash == "ghi789"
        assert metadata.model_config == {"text_model": "test-text"}
        assert metadata.creation_time == datetime(2023, 1, 1, 12, 0, 0)
        assert metadata.file_size == 512
        assert metadata.version == "1.0"

    def test_from_dict_default_version(self):
        """Test creating CacheMetadata from dictionary without version."""
        data = {
            "labels_hash": "jkl012",
            "model_config": {},
            "creation_time": "2023-01-01T12:00:00",
            "file_size": 256,
        }

        metadata = CacheMetadata.from_dict(data)
        assert metadata.version == "1.0"


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_init(self):
        """Test CacheManager initialization."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)
        assert cache_manager.vector_store == vector_store

    def test_compute_labels_hash(self):
        """Test computing hash of labels."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        labels1 = ["T cell", "B cell", "NK cell"]
        labels2 = ["NK cell", "B cell", "T cell"]  # Same labels, different order
        labels3 = ["T cell", "B cell", "Monocyte"]  # Different labels

        hash1 = cache_manager.compute_labels_hash(labels1)
        hash2 = cache_manager.compute_labels_hash(labels2)
        hash3 = cache_manager.compute_labels_hash(labels3)

        # Same labels should have same hash regardless of order
        assert hash1 == hash2
        # Different labels should have different hash
        assert hash1 != hash3
        # Hash should be string of expected length (16 chars)
        assert len(hash1) == 16
        assert isinstance(hash1, str)

    def test_get_cache_metadata_path(self, temp_dir):
        """Test getting cache metadata file path."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "embeddings.npz"
        metadata_path = cache_manager.get_cache_metadata_path(cache_file)

        assert metadata_path == temp_dir / "embeddings.npz.meta"

    def test_save_cache_metadata_success(self, temp_dir):
        """Test successful cache metadata saving."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test_embeddings.npz"
        cache_file.touch()  # Create file
        cache_file.write_text("dummy content")  # Add some content for size

        labels = ["T cell", "B cell"]
        model_config = {"model": "test", "provider": "test"}

        result = cache_manager.save_cache_metadata(cache_file, labels, model_config)

        assert result is True

        # Verify metadata file was created
        metadata_path = cache_file.with_suffix(cache_file.suffix + ".meta")
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "labels_hash" in data
        assert data["model_config"] == model_config
        assert "creation_time" in data
        assert data["file_size"] > 0

    def test_save_cache_metadata_file_not_found(self, temp_dir):
        """Test saving metadata when cache file doesn't exist."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "non_existent.npz"
        labels = ["T cell"]
        model_config = {"model": "test"}

        result = cache_manager.save_cache_metadata(cache_file, labels, model_config)
        assert result is False

    def test_load_cache_metadata_success(self, temp_dir):
        """Test successful cache metadata loading."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test_embeddings.npz"
        cache_file.touch()
        cache_file.write_text("dummy")

        # Save metadata first
        labels = ["T cell", "B cell"]
        model_config = {"model": "test", "provider": "test"}
        cache_manager.save_cache_metadata(cache_file, labels, model_config)

        # Load metadata
        metadata = cache_manager.load_cache_metadata(cache_file)

        assert metadata is not None
        assert isinstance(metadata, CacheMetadata)
        assert metadata.model_config == model_config
        assert len(metadata.labels_hash) == 16

    def test_load_cache_metadata_file_not_found(self, temp_dir):
        """Test loading metadata when metadata file doesn't exist."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "non_existent.npz"
        metadata = cache_manager.load_cache_metadata(cache_file)
        assert metadata is None

    def test_load_cache_metadata_invalid_json(self, temp_dir):
        """Test loading metadata with invalid JSON."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        metadata_file = cache_file.with_suffix(cache_file.suffix + ".meta")

        # Create invalid JSON
        with open(metadata_file, "w") as f:
            f.write("invalid json {")

        metadata = cache_manager.load_cache_metadata(cache_file)
        assert metadata is None

    def test_validate_cache_success(self, temp_dir):
        """Test successful cache validation."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test_embeddings.npz"
        cache_file.touch()
        cache_file.write_text("dummy content")

        labels = ["T cell", "B cell"]
        model_config = {"model": "test", "provider": "test"}

        # Save metadata
        cache_manager.save_cache_metadata(cache_file, labels, model_config)

        # Validate cache
        result = cache_manager.validate_cache(cache_file, labels, model_config)
        assert result is True

    def test_validate_cache_file_not_exists(self, temp_dir):
        """Test cache validation when file doesn't exist."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "non_existent.npz"
        labels = ["T cell"]
        model_config = {"model": "test"}

        result = cache_manager.validate_cache(cache_file, labels, model_config)
        assert result is False

    def test_validate_cache_no_metadata(self, temp_dir):
        """Test cache validation without metadata."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        cache_file.touch()

        labels = ["T cell"]
        model_config = {"model": "test"}

        result = cache_manager.validate_cache(cache_file, labels, model_config)
        assert result is False

    def test_validate_cache_labels_mismatch(self, temp_dir):
        """Test cache validation with mismatched labels."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        cache_file.touch()
        cache_file.write_text("dummy")

        original_labels = ["T cell", "B cell"]
        original_config = {"model": "test"}

        # Save with original labels
        cache_manager.save_cache_metadata(cache_file, original_labels, original_config)

        # Validate with different labels
        new_labels = ["NK cell", "Monocyte"]
        result = cache_manager.validate_cache(cache_file, new_labels, original_config)
        assert result is False

    def test_validate_cache_config_mismatch(self, temp_dir):
        """Test cache validation with mismatched model config."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        cache_file.touch()
        cache_file.write_text("dummy")

        labels = ["T cell", "B cell"]
        original_config = {"model": "original"}

        # Save with original config
        cache_manager.save_cache_metadata(cache_file, labels, original_config)

        # Validate with different config
        new_config = {"model": "different"}
        result = cache_manager.validate_cache(cache_file, labels, new_config)
        assert result is False

    def test_load_cached_embeddings_success(
        self, temp_dir, sample_embeddings, sample_user_labels
    ):
        """Test successful cached embeddings loading."""
        vector_store = Mock(spec=VectorStore)
        vector_store.load_embeddings.return_value = (
            sample_embeddings,
            [f"term:{i}" for i in range(len(sample_embeddings))],
            {"version": "1.0"},
        )

        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        cache_file.touch()
        cache_file.write_text("dummy content")

        model_config = {"model": "test"}

        # Save metadata to make validation pass
        cache_manager.save_cache_metadata(cache_file, sample_user_labels, model_config)

        result = cache_manager.load_cached_embeddings(
            cache_file, sample_user_labels, model_config
        )

        assert result is not None
        embeddings, labels = result
        np.testing.assert_array_equal(embeddings, sample_embeddings)
        assert labels == sample_user_labels

    def test_load_cached_embeddings_invalid_cache(self, temp_dir):
        """Test loading cached embeddings with invalid cache."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        labels = ["T cell"]
        model_config = {"model": "test"}

        # Don't create metadata, so validation will fail
        result = cache_manager.load_cached_embeddings(cache_file, labels, model_config)
        assert result is None

    def test_save_embeddings_with_metadata_success(
        self, temp_dir, sample_embeddings, sample_user_labels
    ):
        """Test successful embeddings and metadata saving."""
        vector_store = Mock(spec=VectorStore)

        def mock_save_embeddings(*args, **kwargs):
            # Create the file when save_embeddings is called
            cache_file = args[2] if len(args) > 2 else kwargs.get("filepath")
            cache_file.touch()
            cache_file.write_text("dummy content")
            return True

        vector_store.save_embeddings.side_effect = mock_save_embeddings

        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        model_config = {"model": "test", "provider": "test"}

        result = cache_manager.save_embeddings_with_metadata(
            sample_embeddings, sample_user_labels, cache_file, model_config
        )

        assert result is True
        vector_store.save_embeddings.assert_called_once()

    def test_save_embeddings_with_metadata_embeddings_fail(
        self, temp_dir, sample_embeddings, sample_user_labels
    ):
        """Test saving embeddings and metadata when embeddings save fails."""
        vector_store = Mock(spec=VectorStore)
        vector_store.save_embeddings.return_value = False

        cache_manager = CacheManager(vector_store)

        cache_file = temp_dir / "test.npz"
        model_config = {"model": "test"}

        result = cache_manager.save_embeddings_with_metadata(
            sample_embeddings, sample_user_labels, cache_file, model_config
        )

        assert result is False

    def test_cleanup_invalid_caches(self, temp_dir):
        """Test cleaning up invalid cache files."""
        vector_store = Mock(spec=VectorStore)

        def mock_load_embeddings(file_path):
            # Return different responses based on which file is being loaded
            if "corrupted.npz" in str(file_path):
                return None  # Can't load corrupted file
            elif "valid.npz" in str(file_path):
                return (np.array([1, 2, 3]), ["id1"], {"version": "1.0"})  # Valid file
            else:
                return None  # Other files fail to load

        vector_store.load_embeddings.side_effect = mock_load_embeddings

        cache_manager = CacheManager(vector_store)

        # Create test cache files
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()

        # File without metadata
        file1 = cache_dir / "no_metadata.npz"
        file1.touch()

        # File with invalid metadata
        file2 = cache_dir / "invalid_metadata.npz"
        file2.touch()
        meta2 = file2.with_suffix(file2.suffix + ".meta")
        with open(meta2, "w") as f:
            f.write("invalid json")

        # File that can't be loaded (corrupted)
        file3 = cache_dir / "corrupted.npz"
        file3.touch()
        meta3 = file3.with_suffix(file3.suffix + ".meta")
        with open(meta3, "w") as f:
            json.dump(
                {
                    "labels_hash": "test",
                    "model_config": {},
                    "creation_time": datetime.now().isoformat(),
                    "file_size": 10,
                    "version": "1.0",
                },
                f,
            )

        # Valid file that should not be removed
        file4 = cache_dir / "valid.npz"
        file4.touch()
        meta4 = file4.with_suffix(file4.suffix + ".meta")
        with open(meta4, "w") as f:
            json.dump(
                {
                    "labels_hash": "test",
                    "model_config": {},
                    "creation_time": datetime.now().isoformat(),
                    "file_size": 10,
                    "version": "1.0",
                },
                f,
            )

        cleaned_count = cache_manager.cleanup_invalid_caches(cache_dir)

        # Should have cleaned 3 files (no_metadata, invalid_metadata, corrupted)
        assert cleaned_count == 3
        assert not file1.exists()
        assert not file2.exists()
        assert not meta2.exists()
        assert not file3.exists()
        assert not meta3.exists()
        assert file4.exists()  # Valid file should remain
        assert meta4.exists()

    def test_cleanup_invalid_caches_no_directory(self, temp_dir):
        """Test cleaning up caches when directory doesn't exist."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        non_existent_dir = temp_dir / "non_existent"
        cleaned_count = cache_manager.cleanup_invalid_caches(non_existent_dir)
        assert cleaned_count == 0

    def test_get_cache_stats(self, temp_dir):
        """Test getting cache statistics."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()

        # Create valid cache file with metadata
        file1 = cache_dir / "valid.npz"
        file1.write_text("A" * 1000)  # 1000 bytes
        meta1 = file1.with_suffix(file1.suffix + ".meta")
        with open(meta1, "w") as f:
            json.dump(
                {
                    "labels_hash": "test",
                    "model_config": {},
                    "creation_time": "2023-01-01T12:00:00",
                    "file_size": 1000,
                    "version": "1.0",
                },
                f,
            )

        # Create invalid cache file without metadata
        file2 = cache_dir / "invalid.npz"
        file2.write_text("B" * 500)  # 500 bytes

        stats = cache_manager.get_cache_stats(cache_dir)

        assert stats["total_files"] == 2
        assert abs(stats["total_size_mb"] - (1500 / (1024 * 1024))) < 0.001  # ~1.43 MB
        assert stats["valid_files"] == 1
        assert stats["invalid_files"] == 1
        assert stats["oldest_cache"] == "2023-01-01T12:00:00"
        assert stats["newest_cache"] == "2023-01-01T12:00:00"

    def test_get_cache_stats_no_directory(self, temp_dir):
        """Test getting cache stats when directory doesn't exist."""
        vector_store = Mock(spec=VectorStore)
        cache_manager = CacheManager(vector_store)

        non_existent_dir = temp_dir / "non_existent"
        stats = cache_manager.get_cache_stats(non_existent_dir)

        expected_stats = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "valid_files": 0,
            "invalid_files": 0,
            "oldest_cache": None,
            "newest_cache": None,
        }
        assert stats == expected_stats
