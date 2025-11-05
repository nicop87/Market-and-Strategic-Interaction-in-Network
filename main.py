import sys, networkx as nx

def load_market(path):
    G = nx.read_gml(path, label='id')
    n = len(G.nodes)//2
    sellers = range(n)
    buyers  = range(n, 2*n)
    valkey = next(k for _,_,d in G.edges(data=True) for k in ("valuation","value","weight") if k in d)
    prices = {j: float(G.nodes[j].get("price",0)) for j in sellers}
    return G, sellers, buyers, n, valkey, prices

def v(G,valkey,i,j): return float(G[i][j][valkey]) if G.has_edge(i,j) else float("-inf")

def demand_graph(G,sellers,buyers,valkey,prices):
    H = nx.Graph()
    for i in buyers:
        best = max((v(G,valkey,i,j)-prices[j] for j in sellers), default=float("-inf"))
        for j in sellers:
            if abs(v(G,valkey,i,j)-prices[j]-best)<1e-9 and best>=0:
                H.add_edge(i,j)
    return H

def constricted(H,buyers,M):
    from collections import deque
    unmatched = [i for i in buyers if i not in M]
    R, S, dq = set(unmatched), set(), deque(unmatched)
    while dq:
        i = dq.popleft()
        for j in H.neighbors(i):
            if j in S: continue
            S.add(j)
            for bi,sj in M.items():
                if sj==j and bi not in R:
                    R.add(bi); dq.append(bi)
    return R,S

def epsilon(G,valkey,prices,R,S,sellers):
    e=float("inf")
    for i in R:
        inS=max(v(G,valkey,i,j)-prices[j] for j in S)
        outS=max(v(G,valkey,i,j)-prices[j] for j in sellers if j not in S)
        if outS>-1e18: e=min(e,inS-outS)
    return e if e>0 else 1e-6

def solve(G,sellers,buyers,n,valkey,prices):
    for r in range(1000):
        H=demand_graph(G,sellers,buyers,valkey,prices)
        M=nx.bipartite.maximum_matching(H,top_nodes=set(buyers))
        M={i:j for i,j in M.items() if i in buyers}
        if len(M)==n: return M,prices
        R,S=constricted(H,buyers,M)
        e=epsilon(G,valkey,prices,R,S,sellers)
        for j in S: prices[j]+=e
    raise RuntimeError("no convergence")

if __name__=="__main__":
    G,sellers,buyers,n,valkey,prices=load_market(sys.argv[1])
    M,prices=solve(G,sellers,buyers,n,valkey,prices)
    print("Final prices:",prices)
    print("Matching:",M)
