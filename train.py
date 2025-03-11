import torch
import numpy as np

def train_model(model, walks, G, window_size=5, num_neg=5, epochs=10, batch_size=32, lr=0.01):
    """Train the SkipGram model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    num_nodes = G.number_of_nodes()
    
    for epoch in range(epochs):
        total_loss = 0
        for walk in walks:
            walk_len = len(walk)
            for i in range(walk_len):
                target = walk[i]
                context = []
                
                # Get context window
                start = max(0, i - window_size)
                end = min(walk_len, i + window_size + 1)
                context += walk[start:i] + walk[i+1:end]
                
                if not context:
                    continue
                
                # Convert to tensors
                target_tensor = torch.LongTensor([target]*len(context)).to(device)
                context_tensor = torch.LongTensor(context).to(device)
                
                # Negative sampling
                neg_context = np.random.choice(num_nodes, size=(len(context), num_neg))
                neg_context = torch.LongTensor(neg_context).to(device)
                
                # Calculate loss
                loss = model(target_tensor, context_tensor, neg_context)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(walks):.4f}")