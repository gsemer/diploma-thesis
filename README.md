# Extendability of Graphs with Perfect Matchings
A matching of a graph is a set of pairwise disjoint edges and it is called perfect if every vertex of the graph is incident to some edge of the matching. The purpose of this thesis is the study of structural and algorithmic properties of graphs with perfect matchings. In particular, we focus on the following question: Assuming that k is a positive integer and G is a graph with perfect matching, is G k-extendable? That is, is it true that for every matching M of cardinality k in G there exists a perfect matching that entirely contains M?\
There is a detailed structural characterization of bipartite graphs G with perfect matchings in terms of the existence of disjoint paths with certain properties which is a direct analogue of Menger's theorem. Let (U,V) be the bipartition of G and M be a perfect matching of G. Graph G is k-extendable if and only if there are k internally disjoint M-alternating paths between every vertex of U and every vertex of V. More strongly, it has been proven that someone can obtain the respective k paths for every other perfect matching M' by using the k paths for a specific perfect matching M.\
From a computational perspective, the Extendability problem focuses on the question whether a graph G is k-extendable or not, where pair (G,k) is the input. The extendability of a graph G, denoted by ext(G), is defined as the maximum k for which G is k-extendable. In the general case, this problem is coNP-complete. In the case where graph G is bipartite, there is a polynomial algorithm that computes ext(G). Thus, the aforementioned problem can be decided in a polynomial amount of time on the number of vertices and edges of G.

### Refferences
---------------
[1] R.E.L. Aldred, D.A. Holton, Dingjun Lou, Akira Saito, M-alternating paths in n-extendable bipartite graphs, Discrete Mathematics 269 (2003), pp. 3-7.

[2] J. Lakhal, L. Litzler, A polynomial algorithm for the extendability problem in bipartite graphs, Information Processing Letters 65 (1998), pp. 11-15.

[3] Dingjun Lou, Akira Saito, Lihua Teng, A note on internally disjoint alternating paths in bipartite graphs, Discrete Mathematics 290 (2005), pp. 105-108.

[4] Jan Hackfeld, Arie M. C. A. Koster, The matching extension problem in general graphs is co-NP-complete, Springer Science+Business Media, LLC, parti of Springer Nature 2017, J Comb Optim (2018), pp. 854-858.
