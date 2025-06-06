import json
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import numpy as np

with open('/Users/leedavis/coding/Python/SupremeCourtGuess/data/basic.json', 'r') as file:
    main = json.load(file)

justice_cases = {
    'John G. Roberts, Jr.': 0,
    'Clarence Thomas': 0,
    'Elena Kagan': 0,
    'Ketanji Brown Jackson': 0,
    'Sonia Sotomayor': 0,
    'Samuel A. Alito, Jr.': 0,
    'Amy Coney Barrett': 0,
    'Neil Gorsuch': 0,
    'Brett M. Kavanaugh': 0
}

G = nx.Graph()
Cases = nx.Graph()
for justice in justice_cases.keys():
    G.add_node(justice)
    Cases.add_node(justice)

for justice in justice_cases.keys():
    for j in justice_cases.keys():
        if justice != j:
            Cases.add_edge(justice, j, weight=0)
            G.add_edge(justice, j, weight=0)

for case in main:
    votes = [vote for vote in case['votes']
             if vote['name'] in justice_cases.keys()]

    for i in range(len(votes)):
        for j in range(i+1, len(votes)):
            Cases[votes[i]['name']][votes[j]['name']]['weight'] += 1

            if votes[i]['vote'] == votes[j]['vote']:
                G[votes[i]['name']][votes[j]['name']]['weight'] += 1

for (u, v, d) in G.edges(data=True):
    if Cases.has_edge(u, v):
        d['weight'] /= Cases[u][v]['weight']

node2vec = Node2Vec(G, dimensions=3, walk_length=30, num_walks=200, workers=4)

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)

embeddings = np.array([model.wv[node] for node in G.nodes()])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the embeddings
ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2])

# Label the nodes
for i, node in enumerate(G.nodes()):
    ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], node)

plt.show()

# pos = nx.spring_layout(G)

# nx.draw(G, pos, with_labels=True, node_color='skyblue',
#         node_size=1500, edge_color='gray')
# plt.show()
