import random
import networkx as nx

def generate_random_walks(G, num_walks=10, walk_length=40):
    """Generate truncated random walks"""
    walks = []
    nodes = list(G.nodes())
    
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            current_node = node
            
            for _ in range(walk_length-1):
                neighbors = list(G.neighbors(current_node))
                if len(neighbors) == 0:
                    break
                current_node = random.choice(neighbors)
                walk.append(current_node)
                
            walks.append(walk)
    return walks