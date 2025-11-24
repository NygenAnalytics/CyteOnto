import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://pub-d8bf3af01ebe421abded39c4cb33d88a.r2.dev/cyteonto"

AVAILABLE_MODELS = {
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

CELL_ONTOLOGY_FILES = {
    "csv": {
        "filename": "cell_to_cell_ontology.csv",
        "url": f"{BASE_URL}/cell_ontology/cell_to_cell_ontology.csv",
    },
    "owl": {"filename": "cl.owl", "url": f"{BASE_URL}/cell_ontology/cl.owl"},
}


def create_directories() -> None:
    """Create necessary directories for downloads."""
    dirs_to_create = [
        "cyteonto/data/embedding/descriptions",
        "cyteonto/data/embedding/cell_ontology",
        "cyteonto/data/cell_ontology",
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file from URL to output path with a custom User-Agent and size display."""
    try:
        print(f"⬇️  Downloading {description or output_path.name}...")
        print(f"\tURL: {url}")
        print(f"\tOutput: {output_path}")

        if output_path.exists():
            print("\t✅ File already exists, skipping download")
            return True

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        req = urllib.request.Request(url, headers=headers)

        with (
            urllib.request.urlopen(req) as response,
            open(output_path, "wb") as out_file,
        ):
            total_size = int(response.info().get("Content-Length", -1))
            chunk_size = 8192
            downloaded = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                # Convert bytes to MB for display
                downloaded_mb = downloaded / (1024 * 1024)

                if total_size > 0:
                    total_mb = total_size / (1024 * 1024)
                    percent = min(100, (downloaded * 100) // total_size)
                    # Updated progress string
                    print(
                        f"\r\tProgress: {downloaded_mb:.2f}/{total_mb:.2f} MB ({percent}%)",
                        end="",
                        flush=True,
                    )
                else:
                    # Fallback for servers that don't provide total size
                    print(
                        f"\r\tProgress: {downloaded_mb:.2f} MB downloaded",
                        end="",
                        flush=True,
                    )

        print("\r\t✅ Downloaded successfully")
        return True

    except urllib.error.URLError as e:
        print(f"\t❌ Failed to download: {e}")
        return False
    except Exception as e:
        print(f"\t❌ Unexpected error: {e}")
        return False


def download_cell_ontology() -> bool:
    """Download cell ontology files."""
    print("\nDownloading Cell Ontology files...")
    success = True

    base_path = Path("cyteonto/data/cell_ontology")

    for file_id, file_info in CELL_ONTOLOGY_FILES.items():
        output_path = base_path / file_info["filename"]
        file_success = download_file(
            file_info["url"], output_path, f"Cell Ontology {file_id.upper()}"
        )
        success = success and file_success

    return success


def download_model_embeddings(model_id: str) -> bool:
    """Download embedding files for a specific model."""
    if model_id not in AVAILABLE_MODELS:
        print(f"❌ Unknown model: {model_id}")
        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return False

    config = AVAILABLE_MODELS[model_id]
    print(f"\nDownloading {config['display_name']} embeddings...")

    success = True

    # Download descriptions
    descriptions_path = Path("cyteonto/data/embedding/descriptions")
    desc_file_path = descriptions_path / str(config["description_file"])
    desc_url = f"{BASE_URL}/descriptions/{config['description_file']}"

    desc_success = download_file(desc_url, desc_file_path, "Descriptions")
    success = success and desc_success

    # Download embeddings
    embeddings_path = Path("cyteonto/data/embedding/cell_ontology")
    emb_file_path = embeddings_path / str(config["embedding_file"])
    emb_url = f"{BASE_URL}/embeddings/{config['embedding_file']}"

    emb_success = download_file(emb_url, emb_file_path, "Embeddings")
    success = success and emb_success

    return success


def main() -> None:
    """Main function to handle command line arguments and downloads."""
    parser = argparse.ArgumentParser(
        description="Download embedding files for CyteOnto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run the following command to see available models:
\tuv run python scripts/show_embeddings.py

Examples:
\tuv run python scripts/download_embedding.py moonshot-ai_kimi-k2
\tuv run python scripts/download_embedding.py deepseek_v3
\tuv run python scripts/download_embedding.py --all
        """,
    )

    parser.add_argument(
        "model_id",
        nargs="?",
        help="Model ID to download",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all available models"
    )
    parser.add_argument(
        "--skip-ontology",
        action="store_true",
        help="Skip downloading cell ontology files",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model_id and not args.all:
        parser.print_help()
        return

    if args.model_id and args.all:
        print("\t❌ Cannot specify both model_id and --all")
        return

    print("CyteOnto Embedding Downloader\n")

    # Create directories
    create_directories()

    overall_success = True

    # Download cell ontology files first (unless skipped)
    if not args.skip_ontology:
        ontology_success = download_cell_ontology()
        overall_success = overall_success and ontology_success

    # Download model embeddings
    if args.all:
        print(f"\n\tDownloading all {len(AVAILABLE_MODELS)} models...")
        for model_id in AVAILABLE_MODELS.keys():
            model_success = download_model_embeddings(model_id)
            overall_success = overall_success and model_success
    else:
        model_success = download_model_embeddings(args.model_id)
        overall_success = overall_success and model_success

    # Summary
    print("\n" + "=" * 50)
    if overall_success:
        print("✅ All downloads completed successfully!")
    else:
        print("⚠️  Some downloads failed. Please check the output above.")
        sys.exit(1)

    print("\nNext steps:")
    print("1. Check the tutorial: notebooks/quick_tutorial.ipynb")


if __name__ == "__main__":
    """
    Script to download embedding files for CyteOnto.

    Usage:
        uv run python scripts/download_embedding.py <model_id>
        uv run python scripts/download_embedding.py --all

    Examples:
        uv run python scripts/download_embedding.py moonshot-ai_kimi-k2
        uv run python scripts/download_embedding.py deepseek_v3
        uv run python scripts/download_embedding.py --all
    """
    main()
