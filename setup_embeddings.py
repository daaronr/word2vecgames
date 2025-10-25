#!/usr/bin/env python3
"""
Setup script to download word embeddings for Word Bocce.

This script uses gensim's API to download pre-trained word embeddings.
The embeddings are saved locally and can be loaded by the game server.

Available models:
1. word2vec-google-news-300 (~1.6GB) - Best quality, large file
2. glove-wiki-gigaword-100 (~130MB) - Smaller, good quality
3. glove-wiki-gigaword-300 (~380MB) - Good balance

For MVP testing, glove-wiki-gigaword-100 is recommended for faster downloads.
"""

import os
import sys
import argparse

try:
    import gensim.downloader as api
except ImportError:
    print("Error: gensim not installed. Please run: pip install gensim")
    sys.exit(1)


MODELS = {
    'google-news': {
        'name': 'word2vec-google-news-300',
        'size': '~1.6GB',
        'description': 'Word2Vec trained on Google News (best quality, large)',
        'dimensions': 300
    },
    'glove-100': {
        'name': 'glove-wiki-gigaword-100',
        'size': '~130MB',
        'description': 'GloVe 100d (small, fast download)',
        'dimensions': 100
    },
    'glove-300': {
        'name': 'glove-wiki-gigaword-300',
        'size': '~380MB',
        'description': 'GloVe 300d (good balance)',
        'dimensions': 300
    }
}


def list_models():
    """List available models."""
    print("\nAvailable embedding models:")
    print("-" * 60)
    for key, info in MODELS.items():
        print(f"\n{key}:")
        print(f"  Name: {info['name']}")
        print(f"  Size: {info['size']}")
        print(f"  Dimensions: {info['dimensions']}")
        print(f"  Description: {info['description']}")
    print()


def download_model(model_key, output_dir='./embeddings'):
    """Download and save a model."""
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print("Use --list to see available models")
        return False

    model_info = MODELS[model_key]
    model_name = model_info['name']

    print(f"\nDownloading {model_name}...")
    print(f"Size: {model_info['size']}")
    print(f"This may take a while depending on your connection...\n")

    try:
        # Download using gensim
        print("Loading model (this downloads if not cached)...")
        model = api.load(model_name)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save in word2vec format
        output_path = os.path.join(output_dir, f"{model_key}.bin")
        print(f"\nSaving to {output_path}...")
        model.save_word2vec_format(output_path, binary=True)

        print(f"\n✓ Success! Model saved to: {output_path}")
        print(f"\nTo use this model, run:")
        print(f"  export MODEL_PATH={os.path.abspath(output_path)}")
        print(f"  uvicorn word_bocce_mvp_fastapi:app --reload")

        return True

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download word embeddings for Word Bocce',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_embeddings.py --list
  python setup_embeddings.py --model glove-100
  python setup_embeddings.py --model google-news --output ./my-embeddings
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODELS.keys()),
        help='Model to download (use --list to see options)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./embeddings',
        help='Output directory for embeddings (default: ./embeddings)'
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model:
        parser.print_help()
        print("\nPlease specify a model with --model, or use --list to see options")
        return

    download_model(args.model, args.output)


if __name__ == '__main__':
    main()
