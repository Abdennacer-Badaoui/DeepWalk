import os
import tarfile
import requests
import numpy as np
import networkx as nx

class GraphDataLoader:
    def __init__(self, data_root='./data'):
        self.data_root = data_root
        self.dataset_urls = {
            'cora': 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
            'citeseer': 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz',
            # 'pubmed': 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz'
        }
        
    def download_and_preprocess(self):
        """Download and preprocess all datasets"""
        os.makedirs(self.data_root, exist_ok=True)
        
        for dataset in self.dataset_urls:
            dataset_dir = self.data_root  
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Download dataset
            print(f'Downloading {dataset}...')
            response = requests.get(self.dataset_urls[dataset], stream=True)
            tar_path = os.path.join(dataset_dir, f'{dataset}.tgz')
            
            with open(tar_path, 'wb') as f:
                f.write(response.content)
            
            # Extract files
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=dataset_dir)
            
            # Remove temporary files
            os.remove(tar_path)
            
            # Dataset-specific processing
            if dataset == 'pubmed':
                self._process_pubmed(dataset_dir)
                
            print(f'{dataset.capitalize()} dataset ready')

    def _process_pubmed(self, dataset_dir):
        """Special processing for PubMed dataset"""
        # Rename files
        os.rename(
            os.path.join(dataset_dir, 'Pubmed-Diabetes.NODE.paper.tab'),
            os.path.join(dataset_dir, 'pubmed.content')
        )
        os.rename(
            os.path.join(dataset_dir, 'Pubmed-Diabetes.DIRECTED.cites.tab'),
            os.path.join(dataset_dir, 'pubmed.cites')
        )
        
        # Clean content file headers
        content_file = os.path.join(dataset_dir, 'pubmed.content')
        with open(content_file, 'r') as f:
            lines = f.readlines()
        
        # Remove header line and description lines
        with open(content_file, 'w') as f:
            f.writelines(lines[1:-1])

    def load_dataset(self, dataset_name):
        """Load processed dataset"""
        dataset_dir = os.path.join(self.data_root, dataset_name)
        
        # Load content file
        content_file = os.path.join(dataset_dir, f'{dataset_name}.content')
        node_map = {}
        features = []
        labels = []
        label_map = {}
        
        with open(content_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                
                # Handle different dataset formats
                if dataset_name == 'pubmed':
                    node_id = parts[0]
                    label = parts[-1].split('=')[-1]
                    feat = parts[1:-1]
                else:
                    node_id = parts[0]
                    label = parts[-1]
                    feat = parts[1:-1]
                
                if label not in label_map:
                    label_map[label] = len(label_map)
                
                labels.append(label_map[label])
                features.append(list(map(float, feat)))
                node_map[node_id] = len(node_map)
        
        # Load citation links
        cites_file = os.path.join(dataset_dir, f'{dataset_name}.cites')
        edges = []
        
        with open(cites_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                
                if dataset_name == 'pubmed':
                    source = parts[1].split(':')[-1]
                    target = parts[3].split(':')[-1]
                else:
                    source = parts[0]
                    target = parts[1]
                
                if source in node_map and target in node_map:
                    edges.append((
                        node_map[source],
                        node_map[target]
                    ))
        
        # Create graph
        G = nx.Graph()
        G.add_edges_from(edges)
        G = nx.convert_node_labels_to_integers(G)
        
        return G, np.array(features), np.array(labels), label_map

    def verify_datasets(self):
        """Verify all datasets are properly loaded"""
        for dataset in self.dataset_urls:
            try:
                G, features, labels, _ = self.load_dataset(dataset)
                print(f"\n{dataset.upper()} Dataset Summary:")
                print(f"Number of nodes: {G.number_of_nodes()}")
                print(f"Number of edges: {G.number_of_edges()}")
                print(f"Feature dimension: {features.shape[1]}")
                print(f"Number of classes: {len(np.unique(labels))}")
            except Exception as e:
                print(f"Error loading {dataset}: {str(e)}")

if __name__ == '__main__':
    loader = GraphDataLoader()
    
    # Download and preprocess datasets
    loader.download_and_preprocess()
    
    # Verify dataset loading
    print("\nVerifying dataset loading...")
    loader.verify_datasets()
    
    # Example usage for single dataset
    print("\nLoading Cora dataset...")
    G, features, labels, label_map = loader.load_dataset('cora')
    print(f"Cora graph contains {G.number_of_nodes()} nodes with {features.shape[1]} features each")