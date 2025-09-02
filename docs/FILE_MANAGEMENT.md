# File Management

## File Organization Structure

CyteOnto automatically organizes your files for optimal caching and study management. Understanding this structure helps you manage your data effectively.

### Overall Directory Structure

```
cyteonto/data/
├── cell_ontology/                              # Core ontology files
│   ├── cl.owl                                  # Cell Ontology OWL file
│   └── cell_to_cell_ontology.csv               # Ontology mappings
├── embedding/                                  # Base embeddings (generated once)
│   ├── cell_ontology/                          # Ontology term embeddings
│   │   ├── embeddings_moonshotai-Kimi-K2-Instruct_Qwen-Qwen3-Embedding-8B.npz
│   │   └── embeddings_model-name_embedding-model.npz
│   └── descriptions/                           # Generated descriptions
│       ├── descriptions_moonshotai-Kimi-K2-Instruct.json
│       └── descriptions_model-name.json
└── user_files/                                 # Study-specific data
    ├── embeddings/                             # User label embeddings
    │   ├── study1/                             # Study-specific organization
    │   │   ├── author/                         # Author labels
    │   │   │   └── author_embeddings_model_embedding-model.npz
    │   │   └── algorithms/                     # Algorithm predictions
    │   │       ├── algorithm1_embeddings_model_embedding-model.npz
    │   │       └── algorithm2_embeddings_model_embedding-model.npz
    │   └── study2/
    └── descriptions/                           # User label descriptions
        ├── study1/
        │   ├── author/
        │   │   └── author_descriptions_model-name.json
        │   └── algorithms/
        │       ├── algorithm1_descriptions_model-name.json
        │       └── algorithm2_descriptions_model-name.json
        └── study2/
```

### File Naming Conventions

#### Base Files (Ontology)
- **Descriptions**: `descriptions_{model-name}.json`
- **Embeddings**: `embeddings_{description-model}_{embedding-model}.npz`

Examples:
- `descriptions_moonshotai-Kimi-K2-Instruct.json`
- `embeddings_moonshotai-Kimi-K2-Instruct_Qwen-Qwen3-Embedding-8B.npz`

#### User Files (Study-specific)
- **Author descriptions**: `author_descriptions_{model-name}.json`
- **Author embeddings**: `author_embeddings_{description-model}_{embedding-model}.npz`
- **Algorithm descriptions**: `{algorithm-name}_descriptions_{model-name}.json`
- **Algorithm embeddings**: `{algorithm-name}_embeddings_{description-model}_{embedding-model}.npz`

Examples:
- `author_descriptions_moonshotai-Kimi-K2-Instruct.json`
- `algorithm1_embeddings_moonshotai-Kimi-K2-Instruct_Qwen-Qwen3-Embedding-8B.npz`

## Caching System
Using caching, we avoid regenerating descriptions and embeddings and ensures reproducible results across runs. CyteOnto automatically validates cache files.

### Cache Management

#### Get Cache Statistics
```python
# Get comprehensive cache information
cache_stats = cyto.get_cache_stats()
print(f"Total cached files: {cache_stats['total_files']}")
print(f"Total cache size: {cache_stats['total_size_mb']:.2f} MB")
print(f"Studies: {cache_stats['studies']}")
print(f"Files by type: {cache_stats['files_by_type']}")
```

#### Clean Up Invalid Cache Files
```python
# Remove corrupted or invalid files
removed_count = cyto.cleanup_user_cache()
print(f"Cleaned up {removed_count} invalid cache files")
```

#### Global Cache Cleanup
```python
# Package-level cache cleanup (use with caution)
import cyteonto

removed_count = cyteonto.cleanup_cache(
    user_data_path="/path/to/user/files"  # Optional custom path
)
print(f"Globally cleaned up {removed_count} files")
```

#### Manual Cache Management
```bash
# View cache structure
tree cyteonto/data/user_files/

# Remove specific study cache
rm -rf cyteonto/data/user_files/embeddings/old_study/
rm -rf cyteonto/data/user_files/descriptions/old_study/

# Check cache sizes
du -sh cyteonto/data/user_files/*/
```

### Cache Configuration

#### Enable/Disable Caching
```python
# Disable caching (not recommended for production)
cyto = cyteonto.CyteOnto(
    base_agent=agent,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    embedding_provider="deepinfra",
    enable_user_file_caching=False
)

# Enable with custom paths
cyto = cyteonto.CyteOnto(
    base_agent=agent,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    embedding_provider="deepinfra",
    base_data_path="/custom/base/path",
    user_data_path="/custom/user/path",
    enable_user_file_caching=True
)
```

## Study Organization

### Study Names as Identifiers

Study names serve as unique identifiers and organize your data:
```python
# Each study creates separate directories
results1 = await cyto.compare_batch(
    author_labels=labels1,
    algo_comparison_data=algo_data1,
    study_name="healthy_liver_2024"  # Creates healthy_liver_2024/
)

results2 = await cyto.compare_batch(
    author_labels=labels2,
    algo_comparison_data=algo_data2,
    study_name="diseased_liver_2024"  # Creates diseased_liver_2024/
)
```

### Best Practices for Study Names
- Use descriptive names: `brain_scrnaseq_2024`, `liver_disease_cohort1`
- Include dates or versions: `study_v1_jan2024`, `rerun_march2024`
- Avoid spaces and special characters: use underscores or hyphens
- Keep names consistent across related analyses
