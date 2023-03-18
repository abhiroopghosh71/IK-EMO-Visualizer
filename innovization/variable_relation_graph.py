import warnings

import networkx as nx
import numpy as np
from innovization.constants import *


class VariableRelationGraph:
    """Encodes the methods and attributes of a Variable Relation Graph (VRG)."""
    def __init__(self, directed=False):
        self.directed = directed  # If True, then the VRG is a directed graph.
        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        self.ranked_node_list = {}
        self.ranked_edge_list = {}

    def add_nodes(self, node_id_list):
        for node_id in node_id_list:
            self.graph.add_node(node_id, label=f'x{node_id}'.translate(sub))

    def add_edges(self, var_pair_list, **kwargs):
        if 'rank' in kwargs:
            rank = kwargs['rank']
        else:
            rank = np.Inf * np.ones(len(var_pair_list))

        if 'rel_type' in kwargs:
            rel_type = kwargs['rel_type']
        else:
            rel_type = -1 * np.ones(len(var_pair_list))

        if 'simple_rel_type' in kwargs:
            simple_rel_type = kwargs['simple_rel_type']
        else:
            simple_rel_type = None

        if 'rel_params' in kwargs:
            rel_params = kwargs['rel_params']
        else:
            rel_params = None

        for i, v in enumerate(var_pair_list):
            self.graph.add_edge(v[0], v[1], rank=rank[i], rel_type=rel_type[i], simple_rel_type=simple_rel_type)

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_edges(self):
        return np.array(self.graph.edges)

    def get_in_edges(self, node_id):
        if self.directed:
            return np.array(list(self.graph.in_edges(node_id)))
        else:
            print("Only directed graphs have incoming edges")

    def get_out_edges(self, node_id):
        if self.directed:
            return np.array(list(self.graph.out_edges(node_id)))
        else:
            print("Only directed graphs have outgoing edges")

    def get_nodes_with_only_incoming_edges(self):
        out_deg = self.graph.out_degree()
        node_list = [n for n in out_deg if out_deg[n] == 0]

        return node_list

    def get_nodes_with_only_outgoing_edges(self):
        in_deg = self.graph.in_degree()
        node_list = [n for n in in_deg if in_deg[n] == 0]

        return node_list

    def remove_nodes_with_degree(self, degree):
        remove_nodes = [node for node, deg in dict(self.graph.degree()).items() if deg == degree]
        self.graph.remove_nodes_from(remove_nodes)

        return remove_nodes

    def transitive_reduction(self):
        if self.directed:
            self.graph = nx.transitive_reduction(self.graph)
        else:
            print("Transitive reduction can only be performed on directed graphs.")

    def draw(self, backend='nx'):
        if backend == 'nx':
            nx.draw(self.graph, with_labels=True)
        else:
            warnings.warn("Unsupported graph drawing backend.")
