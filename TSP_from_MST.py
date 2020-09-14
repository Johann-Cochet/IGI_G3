import tsplib95
from os import linesep

import networkx as nx

import matplotlib.pyplot as plt



"""
Load graph in a networkx Directed Weighted graph

problem contains an TSP instance loaded from tsplib95 format file (using tsplib95 library)
get nodes and arcs (with their weights)
create graph
"""


def TSPlib_to_networkx(problem):
    # Reading nodes
    nodes = list(problem.get_nodes())

    # Reading arcs (and weights)
    arcs = list(problem.get_edges())

    # storing graph on networkx graph

    graph = nx.DiGraph()

    graph.add_nodes_from(nodes)

    arcs_w = []
    for (i, j) in arcs:
        if i != j:  # removing auto-loops
            arcs_w.append((i, j, problem.get_weight(i, j)))

    graph.add_weighted_edges_from(arcs_w)

    return graph


"""Display a graph using spring_layout and node labels"""


def display_graph(graph):
    # position all the nodes
    pos = nx.spring_layout(graph)

    # draw graph element, by element
    # nodes
    nx.draw_networkx_nodes(graph, pos)

    # edges
    nx.draw_networkx_edges(graph, pos)

    # labels
    nx.draw_networkx_labels(graph, pos, font_size=15, font_family='sans-serif')

    plt.show()



def Kruskal(graph):  # graph (undirect) of networkx
    # Variable qui sert à compter le nombre d'arcs dans notre solution pour vérifier la condition d'arrêt
    num_edge = 0
    # Coût de notre MST
    mst_cost = 0
    # On instancie le graphe solution
    mst_graph = nx.create_empty_copy(graph)
    # Variable qui va contenir les arêtes et leurs poids
    arcs_weight = []
    for (u, v, d) in graph.edges(data='weight'):
        arcs_weight.append((u, v, d))
    # On trie la liste "arcs_weight" selon les poids des arcs dans l'ordre croissant
    arcs_weight = sorted(arcs_weight, key=lambda x: x[2])
    # Condition d'arrêt : le nombre d'arcs doit être égal au (nombre de sommets -1)
    while num_edge < len(graph.nodes()) - 1:
        # On stocke dans une variable temporaire l'arc visité à cette étape
        temp = arcs_weight.pop(0)
        # On vérifie s'il n'existe pas déjà un chemin (donc si on risque de créer un cycle)
        if not nx.has_path(mst_graph, temp[0], temp[1]):
            # Si pas de cycle, on rajoute notre arc au graphe solution
            mst_graph.add_edge(*temp[0:2])
            mst_cost += temp[2]
            num_edge += 1
    return (mst_graph, mst_cost)


#----------------------------------------------------------------------------------------------------------
#TSP Heuristic
#----------------------------------------------------------------------------------------------------------

def TSP_from_MST(graph):

    # On transforme notre graphe en graphe non-orienté pour appliquer Kruskal
    orig_graph = graph.to_undirected()

    # On applique l'algorithme de Kruskal pour obtenir un MST
    (mst_graph, mst_value) = Kruskal(orig_graph)

    # On "double" les arêtes du MST
    mst_graph = mst_graph.to_directed()

    # On crée une liste de tous les noeuds du graphe
    nodes = list(mst_graph.nodes)

    # On crée une liste de tous les arcs du graphe
    edges = [e for e in mst_graph.edges]

    nodes_visited = []
    edges_visited = []

    # noeud visité à l'étape i
    noeud_principal = nodes[0]

    # Garde en mémoire les valeurs du "noeud principal"
    historique = []

    # Creating a graph only using the nodes
    tsp_graph = nx.create_empty_copy(graph)

    # Condition d'arrêt : tant qu'on n'a pas visité tous les noeuds
    while len(nodes_visited) < len(nodes) :

        # Variable qui vaut True si on trouve un fils au noeud que l'on visite
        bool = False
        for arc in edges :

            # On s'intéresse aux fils du noeud principal uniquement
            if arc[0] == noeud_principal and (arc[0] not in historique) :

                # On ne rajoute le noeud que s'il n'y est pas pour éviter les doublons
                if arc[0] not in nodes_visited :
                    nodes_visited.append(noeud_principal)
                historique.append(noeud_principal)

                # Le noeud principal prend la valeur de son fils
                noeud_principal = arc[1]

                # On supprime l'arc retenu pour ne pas le reprendre si on reboucle à cette étape
                edges.remove(arc)
                edges.remove(arc[::-1])

                # On change la valeur de bool puisqu'on a trouvé un fils
                bool = True

        if bool == False :
            temp = noeud_principal

            # Le noeud principal reprend sa précédente valeur (on remonte dans la pile)
            noeud_principal = historique.pop()

            # On rajoute le noeud visité s'il n'est pas déjà dans la pile
            if temp not in nodes_visited :
                nodes_visited.append(temp)

    # On crée les arcs du graphe en regroupant les sommets deux à deux
    for i in range(len(nodes_visited)-1) :
        for j in range(i+1,i+2) :
            edges_visited.append((nodes_visited[i],nodes_visited[j]))

    # On rajoute le noeud de départ pour faire une boucle
    edges_visited.append((nodes_visited[-1],nodes[0]))

    tsp_graph.add_edges_from(edges_visited)
    return tsp_graph, edges_visited



def TSP_from_MST2(graph, i = 0) :
    
    # On transforme notre graphe en graphe non-orienté pour appliquer Kruskal
    orig_graph = graph.to_undirected()

    # On applique l'algorithme de Kruskal pour obtenir un MST
    (mst_graph, mst_value) = Kruskal(orig_graph)

    # On "double" les arêtes du MST
    mst_graph = mst_graph.to_directed()

    # On crée une liste de tous les noeuds du graphe
    nodes = list(mst_graph.nodes)

    # On crée une liste de tous les arcs du graphe
    edges = [e for e in mst_graph.edges]

    # ensemble des noeuds et des arcs visités
    nodes_visited = set(nodes[i])
    edges_visited = set()

    # pile des noeuds avec le noeud de départ comme premier élément
    path_nodes = [nodes[i]]
    
    
    def path (graph, nodes_visited, edges_visited, path_nodes) :
        if len(path_nodes) == 0 :
            return edges_visited
        else :
            neighbors = list(graph.neighbors(path_nodes[-1]))
            for i in neighbors :
                if i not in nodes_visited :
                    nodes_visited.add(i)
                    edges_visited.add((path_nodes[-1], i))
                    return path (graph, nodes_visited, edges_visited, path_nodes + [i])
                
        
    

# -------------------------------------------------------------------------------------------------

# Loading a TSP problem
#-------------------------------------------------------------------------------------------------------
# Work on berlin52:
#----------------------------------------------------------------------------------------------------
# problem = tsplib95.load_problem('./TSPData/berlin52.tsp')


#----------------------------------------------------------------------------------------------------
# Work on berlin10:
#----------------------------------------------------------------------------------------------------
problem = tsplib95.load_problem('./TSPData/berlin10.tsp')


#----------------------------------------------------------------------------------------------------
# Work on a280 :
#----------------------------------------------------------------------------------------------------
# problem = tsplib95.load_problem('./TSPData/a280.tsp')



#-----------------------------------------------------------------------------------------------------
# Solution TSP
#------------------------------------------------------------------------------------------------------
orig_graph = TSPlib_to_networkx(problem)

tsp_solution, edges_d = TSP_from_MST(orig_graph)

print(problem.name)
print(problem.type)
print(problem.dimension)

display_graph(tsp_solution)
