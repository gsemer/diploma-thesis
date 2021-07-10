from hopcroftkarp import HopcroftKarp
import collections
import sys

# this is a class of an undirected graph
# its purpose is the following:
# i) check if the graph is connected
# ii) check if the graph is bipartite
# iii) check if the graph contains at least one perfect matching
# iv) find a maximum matching


class Undirected:

    def __init__(self, red, blue):
        self.red = red
        self.blue = blue

    # a graph G is connected if and only if there is a path between every pair of two distinct vertices
    @staticmethod
    def isConnected(dictionary):

        # this is a function to check if there is a path between start and end node
        def reachable(start, end):
            # BFS procedure
            # mark all the vertices as not visited
            visited = {}
            for w in dictionary.keys():
                visited[w] = False
            # initialize a queue for BFS and put start node into it
            queue = [start]
            # mark the start node as visited and enqueue it
            visited[start] = True
            # while queue is not empty, do
            while queue:
                # de-queue a vertex from queue
                vertex = queue.pop(0)
                # if this adjacent node is the destination node, then there is a path in the graph
                if vertex == end:
                    return True
                # otherwise, continue to do BFS
                for neighbor in dictionary[vertex]:
                    # check if the neighbor of vertex is not visited
                    if not visited[neighbor]:
                        # mark the neighbor node as visited and enqueue it
                        queue.append(neighbor)
                        visited[neighbor] = True
            # if BFS is complete without end node is marked as visited
            return False

        for u in dictionary.keys():
            for v in dictionary.keys():
                if u != v:
                    # if there is no path between u and v, then the graph is not connected
                    # check if there is not a path between u and v in the graph
                    if not reachable(u, v):
                        return False
        # if for loop is complete, then the graph is connected
        return True

    # a graph G is bipartite if and only if the graph admits two colors
    def isBipartite(self, dictionary):

        # this is a function to check if the vertices can admit two colors
        def admitsTwoColors(node):
            # BFS procedure
            # mark all the vertices as not colored
            colored = {}
            for w in dictionary.keys():
                colored[w] = ''
            # initialize a queue for BFS and put node into it
            queue = [node]
            # color the input node
            colored[node] = 'RED'
            # while queue is not empty, do
            while queue:
                # de-queue a vertex from queue
                vertex = queue.pop(0)
                for neighbor in dictionary[vertex]:
                    # check if the neighbor of vertex is not colored
                    if colored[neighbor] == '':
                        if colored[vertex] == 'RED':
                            colored[neighbor] = 'BLUE'
                        if colored[vertex] == 'BLUE':
                            colored[neighbor] = 'RED'
                        queue.append(neighbor)
                    # check if two adjacent vertices share the same color
                    # if yes, then reject
                    if colored[vertex] == colored[neighbor]:
                        return False
            # store the vertices into two lists according to their color
            for w in dictionary.keys():
                if colored[w] == 'RED':
                    self.red.append(w)
                if colored[w] == 'BLUE':
                    self.blue.append(w)
            # if BFS is complete without finding two adjacent nodes with same color, then accept
            return True

        # select the first vertex in graph
        v = [w for w in list(dictionary.keys())][0]
        # if the graph admits 2 colors, then accept
        if admitsTwoColors(v):
            return True
        # otherwise, reject
        else:
            return False

    # check if the graph contains a perfect matching by finding a maximum matching
    def hasPerfectMatching(self, dictionary):
        # if the two subsets do not share the same number of elements, then reject
        if len(self.red) != len(self.blue):
            return False
        # find a maximum matching of the graph
        temporary = {}
        for vertex in self.red:
            temporary[vertex] = set(dictionary[vertex])
        matching_dictionary = HopcroftKarp(temporary).maximum_matching()
        matching = []
        for vertex in matching_dictionary.keys():
            for neighbor in matching_dictionary[vertex]:
                if not (neighbor, vertex) in matching:
                    matching.append((vertex, neighbor))
        # check if the length of the matching is equal to the number of nodes divided by 2
        # if yes, then accept
        if len(matching) == len(list(dictionary.keys())) / 2:
            return True
        # otherwise, reject
        return False

    # find a maximum matching of the graph
    def findMaximumMatching(self, dictionary):
        temporary = {}
        for vertex in self.red:
            temporary[vertex] = set(dictionary[vertex])
        matching_dictionary = HopcroftKarp(temporary).maximum_matching()
        matching = []
        for vertex in matching_dictionary.keys():
            for neighbor in matching_dictionary[vertex]:
                if not (neighbor, vertex) in matching:
                    matching.append((vertex, neighbor))
        return matching


# this is a class of a directed graph
# it inherits everything from the above class
# its purpose is the following:
# i) convert the undirected graph into directed according to a specific principle
# ii) check if the graph is strongly connected
# iii) find the maximum of the maximum number of vertex disjoint paths from every RED vertex to every BLUE vertex


class Directed(Undirected):

    def __init__(self, red, blue):
        Undirected.__init__(self, red, blue)

    # convert the undirected graph into directed according to the following property
    # an edge is called saturated if it belongs to the matching
    # an edge is called unsaturated if it does not belong to the matching
    # direct the saturated edges from BLUE to RED
    # direct the unsaturated edges from RED to BLUE
    def direct(self, dictionary, matching):

        def generateEdges():
            edges = []
            for vertex in dictionary.keys():
                for neighbor in dictionary[vertex]:
                    if (neighbor, vertex) not in edges:
                        edges.append((vertex, neighbor))
            return edges

        directed_edges = generateEdges()
        # every edge must have in its first coordinate a RED vertex
        for i in range(len(directed_edges)):
            if directed_edges[i][0] in self.blue and directed_edges[i][1] in self.red:
                directed_edges[i] = (directed_edges[i][1], directed_edges[i][0])
        # do the same for matching
        for i in range(len(matching)):
            if matching[i][0] in self.blue and matching[i][1] in self.red:
                matching[i] = (matching[i][1], matching[i][0])
        # initialize the directed graph
        directed = {}
        for w in dictionary.keys():
            directed[w] = []
        # proceed with the aforementioned methodology
        for edge in directed_edges:
            if edge in matching:
                directed[edge[1]].append(edge[0])
            else:
                directed[edge[0]].append(edge[1])
        return directed

    # check if the directed graph is strongly connected
    def isStronglyConnected(self, directed):
        # by construction of the dictionary it suffices to check if the graph is connected
        if self.isConnected(directed):
            return True
        else:
            return False

    # let G be the obtained digraph.
    # create a new digraph G' obtained from G as follows:
    # i) for each u except source and sink, create two vertices u_in and u_out and an edge (u_in, u_out)
    # ii) for each (source, y), create the edge (source, y_in)
    # iii) for each (x, source), create the edge (x_out, source)
    # iv) for each (x, sink), create the edge (x_out, sink)
    # v) for each (sink, y), create the edge (sink, y_in)
    # vi) for each (x, y), create the edge (x_out, y_in)
    # observe that maximum number of vertex disjoint paths of G equals to maximum number of edge disjoint paths of G'
    # create a new digraph G'' obtained from G' by giving capacity 1 to every edge of G'
    # observe that maximum number of edge disjoint paths of G' equals to the maximum flow of G''
    # thus, it suffices to find the maximum flow of G''
    def maxVertexDisjointPaths(self, directed):

        def maxEdgeDisjointPaths(source_node, sink_node):

            def maximumFlow(matrix, source_pos, sink_pos):

                def BFS(start, end):
                    # mark all the vertices as not visited
                    visited = [False] * len(matrix)
                    # create a queue for BFS
                    queue = collections.deque()
                    # mark the source node as visited and enqueue it
                    queue.append(start)
                    visited[start] = True
                    # standard BFS loop
                    while queue:
                        w = queue.popleft()
                        # get all adjacent vertices of the de-queued vertex u
                        # if an adjacent has not been visited, then mark it as visited and enqueue it
                        for ind, val in enumerate(matrix[w]):
                            if not visited[ind] and (val > 0):
                                queue.append(ind)
                                visited[ind] = True
                                parent[ind] = w
                    # if we reached sink in BFS starting from source, then return true, else false
                    return visited[end]

                # this array is filled by BFS and to store path
                parent = [-1] * len(matrix)
                # there is no flow initially
                max_flow = 0
                # augment the flow while there is path from source to sink
                while BFS(source_pos, sink_pos):
                    # find minimum residual capacity of the edges along the path filled by BFS.
                    # or we can say find the maximum flow through the path found.
                    path_flow = float("Inf")
                    s = sink_pos
                    while s != source_pos:
                        path_flow = min(path_flow, matrix[parent[s]][s])
                        s = parent[s]
                    # add path flow to overall flow
                    max_flow += path_flow
                    # update residual capacities of the edges and reverse edges along the path
                    v = sink_pos
                    while v != source_pos:
                        u = parent[v]
                        matrix[u][v] -= path_flow
                        matrix[v][u] += path_flow
                        v = parent[v]
                return max_flow

            # create an adjacency matrix representation of the graph
            # specifically, the capacity of every edge equals to 1
            edges = {}
            for vertex in dictionary.keys():
                neighbors = []
                for adjacent in dictionary.keys():
                    if adjacent in dictionary[vertex]:
                        neighbors.append(1)
                    else:
                        neighbors.append(0)
                edges[vertex] = neighbors
            adjacency_matrix = []
            for vertex in edges.keys():
                adjacency_matrix.append(edges[vertex])
            # find the position of source node
            source_position = 0
            # find the position of sink node
            sink_position = 0
            # break before run all nodes if possible
            count = 0
            for i in range(len(list(dictionary.keys()))):
                if list(dictionary.keys())[i] == source_node:
                    source_position += i
                    count += 1
                if list(dictionary.keys())[i] == sink_node:
                    sink_position += i
                    count += 1
                if count == 2:
                    break
            # matrix will be used to find the maximum flow of the graph
            edge_disjoint_paths = maximumFlow(adjacency_matrix, source_position, sink_position)
            return edge_disjoint_paths

        # initialize a variable to store number of vertex-disjoint directed paths between every vertex of RED and BLUE
        maximum = float("Inf")
        for source in self.red:
            for sink in self.blue:
                # convert the instance of vertex disjoint paths problem into an instance of edge disjoint paths
                dictionary = {}
                for x in directed.keys():
                    if x != source and x != sink:
                        dictionary[x + '_in'] = []
                        dictionary[x + '_out'] = []
                    else:
                        dictionary[x] = []
                for x in dictionary.keys():
                    if x == source:
                        for neighbor in directed[x]:
                            if neighbor == sink:
                                dictionary[x].append(neighbor)
                            else:
                                dictionary[x].append(neighbor + '_in')
                    elif x == sink:
                        for neighbor in directed[x]:
                            if neighbor == source:
                                dictionary[x].append(neighbor)
                            else:
                                dictionary[x].append(neighbor + '_in')
                    else:
                        if 'in' in x.split('_'):
                            dictionary[x].append(x.split('_')[0] + '_out')
                        else:
                            for neighbor in directed[x.split('_')[0]]:
                                if neighbor == source or neighbor == sink:
                                    dictionary[x].append(neighbor)
                                else:
                                    dictionary[x].append(neighbor + '_in')
                # calculate the number of edge-disjoint paths between source and sink
                vertex_disjoint_paths = maxEdgeDisjointPaths(source, sink)
                # store the maximum so far
                maximum = min(maximum, vertex_disjoint_paths)
        return maximum


def find_extendability(dictionary):
    undirected_graph = Undirected([], [])
    # check if the graph is not connected
    if not undirected_graph.isConnected(dictionary):
        sys.exit('NotConnectedError: Invalid input!')
    # check if the graph is not bipartite
    if not undirected_graph.isBipartite(dictionary):
        sys.exit('NotBipartiteError: Invalid input!')
    else:
        # if the graph is bipartite, then store the two vertex disjoint sets
        red = undirected_graph.red
        blue = undirected_graph.blue
    # check if the graph does not contain a perfect matching
    if not undirected_graph.hasPerfectMatching(dictionary):
        sys.exit('PerfectMatchingNotFoundError: Invalid input!')
    else:
        # find a maximum matching
        # since the graph has a perfect matching, this matching is actually perfect
        matching = undirected_graph.findMaximumMatching(dictionary)
    # initialize a variable for the extendability
    k = 0
    # obtain the directed version of the graph
    directed = Directed(red, blue)
    directed_graph = directed.direct(dictionary, matching)
    # check if the graph is strongly connected
    if directed.isStronglyConnected(directed_graph):
        # calculate the extendability
        k = directed.maxVertexDisjointPaths(directed_graph)
        return 'The graph is {}-extendable'.format(str(k))
    else:
        return 'The graph is not strongly connected. Thus, it is {}-extendable'.format(str(k))


# 2-extendable graph
dictionary1 = {'1': ['5', '6', '7'],
               '2': ['5', '6', '8'],
               '3': ['6', '7', '8'],
               '4': ['5', '7', '8'],
               '5': ['1', '2', '4'],
               '6': ['1', '2', '3'],
               '7': ['1', '3', '4'],
               '8': ['2', '3', '4']}

# 3-extendable graph
dictionary2 = {'1': ['6', '7', '8', '9'],
               '2': ['6', '7', '8', 'a'],
               '3': ['6', '8', '9', 'a'],
               '4': ['7', '8', '9', 'a'],
               '5': ['6', '7', '9', 'a'],
               '6': ['1', '2', '3', '5'],
               '7': ['1', '2', '4', '5'],
               '8': ['1', '2', '3', '4'],
               '9': ['1', '3', '4', '5'],
               'a': ['2', '3', '4', '5']}

# 4-extendable
dictionary3 = {'a': ['1', '2', '3', '4', '5'],
               'b': ['1', '2', '3', '4', '7'],
               'c': ['1', '2', '3', '4', '6'],
               'd': ['1', '2', '5', '6', '7'],
               'e': ['1', '3', '5', '6', '7'],
               'f': ['2', '4', '5', '6', '7'],
               'g': ['3', '4', '5', '6', '7'],
               '1': ['a', 'b', 'c', 'd', 'e'],
               '2': ['a', 'b', 'c', 'd', 'f'],
               '3': ['a', 'b', 'c', 'e', 'g'],
               '4': ['a', 'b', 'c', 'f', 'g'],
               '5': ['a', 'd', 'e', 'f', 'g'],
               '6': ['c', 'd', 'e', 'f', 'g'],
               '7': ['b', 'd', 'e', 'f', 'g']}

# 5-extendable graph
dictionary4 = {'a': ['1', '2', '3', '4', '5', '6'],
               'b': ['1', '2', '3', '4', '5', '7'],
               'c': ['1', '2', '3', '4', '6', '7'],
               'd': ['1', '2', '3', '5', '6', '7'],
               'e': ['1', '2', '4', '5', '6', '7'],
               'f': ['1', '3', '4', '5', '6', '7'],
               'g': ['2', '3', '4', '5', '6', '7'],
               '1': ['a', 'b', 'c', 'd', 'e', 'f'],
               '2': ['a', 'b', 'c', 'd', 'e', 'g'],
               '3': ['a', 'b', 'c', 'd', 'f', 'g'],
               '4': ['a', 'b', 'c', 'e', 'f', 'g'],
               '5': ['a', 'b', 'd', 'e', 'f', 'g'],
               '6': ['a', 'c', 'd', 'e', 'f', 'g'],
               '7': ['b', 'c', 'd', 'e', 'f', 'g']}

# 1-extendable graph
dictionary5 = {'2': ['7', '8'],
               '3': ['8', '9'],
               '4': ['7', '8', '9'],
               '7': ['2', '4'],
               '8': ['2', '3', '4'],
               '9': ['3', '4']}

ext = find_extendability(dictionary1)
print(ext)

