import argparse
import os
import numpy as np
from random_walk import generate_random_walks
from model import SkipGram
from train import train_model
from evaluate import evaluate
from data_loader import GraphDataLoader  # Using the new data loader

def main():
    parser = argparse.ArgumentParser(description='DeepWalk Implementation')
    parser.add_argument('--dataset', type=str, default='cora', 
                       choices=['cora', 'citeseer', 'pubmed'],
                       help='Dataset name')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--num_walks', type=int, default=10,
                       help='Number of walks per node')
    parser.add_argument('--walk_length', type=int, default=40,
                       help='Length of each walk')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Context window size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs')
    parser.add_argument('--download', action='store_true',
                       help='Force download datasets')
    args = parser.parse_args()

    # Initialize data loader
    loader = GraphDataLoader()
    
    # Download datasets if needed
    if args.download or not os.path.exists(f"./data"):
        print(f"Downloading and preprocessing {args.dataset} dataset...")
        loader.download_and_preprocess()

    # Load dataset
    try:
        G, features, labels, label_map = loader.load_dataset(args.dataset)
        print(f"\nLoaded {args.dataset.upper()} dataset:")
        print(f"- Nodes: {G.number_of_nodes()}")
        print(f"- Edges: {G.number_of_edges()}")
        print(f"- Features shape: {features.shape}")
        print(f"- Classes: {len(label_map)}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Generate random walks
    print("\nGenerating random walks...")
    walks = generate_random_walks(G, args.num_walks, args.walk_length)
    print(f"Generated {len(walks)} total walks")
    print(f"Average walk length: {np.mean([len(w) for w in walks]):.1f}")

    # Initialize and train model
    print("\nInitializing model...")
    model = SkipGram(G.number_of_nodes(), args.embed_dim)
    
    print(f"Starting training for {args.epochs} epochs...")
    train_model(model, walks, G, 
               window_size=args.window_size, 
               epochs=args.epochs)

    # Get embeddings and evaluate
    print("\nEvaluating embeddings...")
    embeddings = model.get_embeddings().cpu().numpy()
    results = evaluate(embeddings, labels)
    
    print(f"\nFinal Results for {args.dataset.upper()}:")
    print(f"- Node Classification Accuracy: {results['accuracy']:.4f}")
    print(f"- Silhouette Score: {results['silhouette']:.4f}")

if __name__ == "__main__":
    main()