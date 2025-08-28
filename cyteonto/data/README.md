# Cell Ontology Data files

Place the `cell_to_cell_ontology.csv` and `cl.owl` file in `cell_ontology` folder. 

An `embedding` and `user_files` folder will be created duing runtime.

Place the downloaded cell ontology description and embedding files in `embedding/descriptions` and `embedding/cell_ontology` respectively.

Folder structure

```bash
data
├── cell_ontology
│  ├── cell_to_cell_ontology.csv
│  └── cl.owl
├── embedding
│  ├── cell_ontology
│  └── descriptions
├── README.md
└── user_files
```

The `data` folder at a later stage will look something like:

```bash
data
├── cell_ontology
│  ├── cell_to_cell_ontology.csv
│  ├── cell_to_cell_ontology_sample.csv
│  └── cl.owl
├── embedding
│  ├── cell_ontology
│  │  └── embeddings_Qwen-Qwen3-235B-A22B-Instruct-2507_Qwen-Qwen3-Embedding-8B.npz
│  └── descriptions
│     └── descriptions_Qwen-Qwen3-235B-A22B-Instruct-2507.json
├── README.md
└── user_files
   ├── descriptions
   │  └── dataset_liver_samples
   │     ├── algorithms
   │     │  ├── Algorithm1_descriptions_Qwen-Qwen3-235B-A22B-Instruct-2507.json
   │     │  └── Algorithm2_descriptions_Qwen-Qwen3-235B-A22B-Instruct-2507.json
   │     └── author
   │        └── author_descriptions_Qwen-Qwen3-235B-A22B-Instruct-2507.json
   └── embeddings
      └── dataset_liver_samples
         ├── algorithms
         │  ├── Algorithm1_embeddings_Qwen-Qwen3-235B-A22B-Instruct-2507_Qwen-Qwen3-Embedding-8B.npz
         │  ├── Algorithm1_embeddings_Qwen-Qwen3-235B-A22B-Instruct-2507_Qwen-Qwen3-Embedding-8B.npz.meta
         │  ├── Algorithm2_embeddings_Qwen-Qwen3-235B-A22B-Instruct-2507_Qwen-Qwen3-Embedding-8B.npz
         │  └── Algorithm2_embeddings_Qwen-Qwen3-235B-A22B-Instruct-2507_Qwen-Qwen3-Embedding-8B.npz.meta
         └── author
            ├── author_embeddings_Qwen-Qwen3-235B-A22B-Instruct-2507_Qwen-Qwen3-Embedding-8B.npz
            └── author_embeddings_Qwen-Qwen3-235B-A22B-Instruct-2507_Qwen-Qwen3-Embedding-8B.npz.meta
```
