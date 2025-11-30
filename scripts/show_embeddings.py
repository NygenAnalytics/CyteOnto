from pathlib import Path

AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": {
        "display_name": "LLaMA 3.3 70B Versatile (descriptions) + Qwen3-Embedding-8B (embeddings)",
        "description_file": "descriptions_llama-3.3-70b-versatile.json",
        "embedding_file": "embeddings_llama-3.3-70b-versatile_Qwen-Qwen3-Embedding-8B.npz",
        "recommended": False,
    },
    "moonshot-ai_kimi-k2": {
        "display_name": "Moonshot AI Kimi-K2 (descriptions) + Qwen3-Embedding-8B (embeddings)",
        "description_file": "descriptions_moonshotai-Kimi-K2-Instruct.json",
        "embedding_file": "embeddings_moonshotai-Kimi-K2-Instruct_Qwen-Qwen3-Embedding-8B.npz",
        "recommended": True,
    },
    "deepseek_v3": {
        "display_name": "DeepSeek V3 (descriptions) + Qwen3-Embedding-8B (embeddings)",
        "description_file": "descriptions_deepseek-ai-DeepSeek-V3.json",
        "embedding_file": "embeddings_deepseek-ai-DeepSeek-V3_Qwen-Qwen3-Embedding-8B.npz",
        "recommended": False,
    },
    # "deepseek_v3.1": {
    #     "display_name": "DeepSeek V3.1 descriptions + Qwen3-Embedding-8B embeddings",
    #     "description_file": "descriptions_deepseek-ai-DeepSeek-V3.1.json",
    #     "embedding_file": "embeddings_deepseek-ai-DeepSeek-V3.1_Qwen-Qwen3-Embedding-8B.npz",
    #     "recommended": False,
    # },
}


def check_existing_files() -> dict[str, dict[str, bool]]:
    """Check which embedding files already exist locally."""
    base_path = Path("cyteonto/data")
    descriptions_path = base_path / "embedding" / "descriptions"
    embeddings_path = base_path / "embedding" / "cell_ontology"

    existing = {}
    for model_id, config in AVAILABLE_MODELS.items():
        desc_file = descriptions_path / config["description_file"]  # type: ignore
        emb_file = embeddings_path / config["embedding_file"]  # type: ignore

        existing[model_id] = {
            "description_exists": desc_file.exists(),
            "embedding_exists": emb_file.exists(),
            "both_exist": desc_file.exists() and emb_file.exists(),
        }

    return existing


def main() -> None:
    """Main function to display available embedding models."""
    print("CyteOnto - Available Embedding Models\n")

    existing_files = check_existing_files()

    for model_id, config in AVAILABLE_MODELS.items():
        status_symbol = "✅" if existing_files[model_id]["both_exist"] else "⬇️ "
        recommended_text = " (Recommended)" if config["recommended"] else ""

        print(f"{status_symbol} {model_id}{recommended_text}")
        print(f"    Name: {config['display_name']}")

        if existing_files[model_id]["both_exist"]:
            print("    Status: Already downloaded")
        elif (
            existing_files[model_id]["description_exists"]
            or existing_files[model_id]["embedding_exists"]
        ):
            missing = []
            if not existing_files[model_id]["description_exists"]:
                missing.append("descriptions")
            if not existing_files[model_id]["embedding_exists"]:
                missing.append("embeddings")
            print(f"    Status: Partial - missing {', '.join(missing)}")
        else:
            print("    Status: Not downloaded")

        print()

    # Check cell ontology files
    base_path = Path("cyteonto/data")
    cell_ontology_path = base_path / "cell_ontology"
    csv_file = cell_ontology_path / "cell_to_cell_ontology.csv"
    owl_file = cell_ontology_path / "cl.owl"

    ontology_status = "✅" if csv_file.exists() and owl_file.exists() else "⬇️"
    print(f"{ontology_status} Cell Ontology Files")
    if csv_file.exists() and owl_file.exists():
        print("    Status: Already downloaded")
    else:
        missing = []
        if not csv_file.exists():
            missing.append("CSV mapping")
        if not owl_file.exists():
            missing.append("OWL file")
        print(f"    Status: Missing {', '.join(missing)}")

    print("\nUsage:")
    print("    uv run python scripts/download_embedding.py <model_id>")
    print("    uv run python scripts/download_embedding.py --all")
    print("\nExample:")
    print("    uv run python scripts/download_embedding.py moonshot-ai_kimi-k2")


if __name__ == "__main__":
    """
    Script to show available embedding models for download.

    Usage:
        uv run python scripts/show_embeddings.py
    """
    main()
