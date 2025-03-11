import torch
import torch.nn as nn

class SkipGram(nn.Module):
    """Skip-gram model with negative sampling"""
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with uniform distribution"""
        init_range = 0.5 / self.embeddings.embedding_dim
        nn.init.uniform_(self.embeddings.weight, -init_range, init_range)
        
    def forward(self, target, context, neg_context):
        """Calculate loss for positive and negative samples"""
        emb_target = self.embeddings(target)
        emb_context = self.embeddings(context)
        emb_neg = self.embeddings(neg_context)
        
        # Positive score
        pos_score = torch.sum(emb_target * emb_context, dim=1)
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_score) + 1e-15))
        
        # Negative score
        neg_score = torch.sum(emb_target.unsqueeze(1) * emb_neg, dim=2)
        neg_loss = -torch.mean(torch.log(torch.sigmoid(-neg_score) + 1e-15))
        
        return pos_loss + neg_loss
    
    def get_embeddings(self):
        return self.embeddings.weight.data
