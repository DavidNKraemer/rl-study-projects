import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import write_dot


def get_mdp_graph(mdp):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(mdp.states)
    for state in mdp.states:
        for next_state in mdp.states:
            for a, action in enumerate(mdp.actions):
                if mdp.tp(state, action, next_state) > 0:
                    graph.add_edge(state, next_state, key=(state, a, next_state),
                                   action=f"{action}",
                                   reward=mdp.rewards(state, action),
                                   prob=mdp.tp(state, action, next_state)
                                   )

    return graph

def set_graph_policy(graph, mdp, policy):
    for state, next_state, (_, action, _) in graph.edges:
        if policy[state] == action:
            graph.edges[state, next_state]['color'] = 'red'

def draw(graph):
    nx.draw(graph)
    plt.show()

