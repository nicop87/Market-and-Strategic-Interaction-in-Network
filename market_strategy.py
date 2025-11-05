
import argparse
import math
import sys
import os
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


# Matplotlib is optional unless --plot is passed
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

# Market class for easier variable transport between functions
@dataclass
class Market:
    G: nx.Graph
    sellers: List[int]
    buyers: List[int]
    n: int
    prices: Dict[int, float]
    valkey: str

# loads the market from the gml and formats it into the Market Class object
def load_market(path: str) -> Market:
    # Checks if the file exists and opens
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        G = nx.read_gml(path, label='id')
    except Exception as e:
        raise ValueError(f"Failed to read GML: {e}")

    # Makes sure it is undirected graph object
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G.to_undirected())

    # Following checks that input graph is valid along constraints and format
    try:
        node_ids = sorted(int(u) for u in G.nodes())
    except Exception:
        raise ValueError("Node identifiers must be integers 0..2n-1 in the GML file.")
    if not node_ids:
        raise ValueError("Graph has no nodes.")
    if node_ids[0] != 0:
        raise ValueError("Smallest node id must be 0.")
    m= len(G.nodes)# node_ids[-1] 
    if m % 2 != 0:
        raise ValueError("Number of nodes must be even (2n).")
    n = m // 2
    expected_ids = list(range(0, 2 * n))
    if node_ids != expected_ids:
        raise ValueError("Node ids must be exactly 0..(2n-1) with no gaps.")

    # Organizes our sellers and buyers in lists
    sellers = list(range(0, m // 2))
    buyers = list(range(m // 2, m))

    # Initialize prices from seller node attributes, default 0.0
    prices: Dict[int, float] = {}
    for a in sellers:
        attr = G.nodes[a]
        p = float(attr.get("price", 0.0))
        if math.isnan(p) or math.isinf(p) or p < 0:
            raise ValueError(f"Invalid price on seller node {a}: {p}")
        prices[a] = p

    return Market(G=G, sellers=sellers, buyers=buyers, n=n, prices=prices, valkey="valuation")

# just grabs the valuation of the edge
def valuation(mkt: Market, buyer: int, seller: int) -> float:
    data = mkt.G.get_edge_data(buyer, seller, default=None)
    val = data.get(mkt.valkey, None)
    return float(val)

# Returns a graph with only the preferred edges on display
# Returns a dictionary that maps each buyer to it's preferred seller
def build_demand_graph(mkt: Market) -> Tuple[nx.Graph, Dict[int, float]]:
    H = nx.Graph()
    H.add_nodes_from(mkt.buyers, bipartite=0)
    H.add_nodes_from(mkt.sellers, bipartite=1)

    max_surplus: Dict[int, float] = {}
    for i in mkt.buyers:
        best = float("-inf")
        best_sellers: List[int] = []
        for j in mkt.sellers:
            vij = valuation(mkt, i, j)
            if vij == float("-inf"):
                continue
            s = vij - mkt.prices[j]
            if s > best + 1e-12:
                best = s
                best_sellers = [j]
            elif abs(s - best) <= 1e-12:
                best_sellers.append(j)
        if best >= -1e-12:  # allow tiny negative to count as zero
            kept = []
            for j in best_sellers:
                if valuation(mkt, i, j) != float("-inf"):
                    H.add_edge(i, j)
                    kept.append(j)
            max_surplus[i] = max(0.0, best)
        else:
            max_surplus[i] = best  # negative; no edges added for this buyer
    return H, max_surplus

# Returns list of unique sellers from preferred graph
def maximum_matching_on_demand(H: nx.Graph, buyers: List[int]) -> Dict[int, int]:
    """Return matching as mapping buyer->seller for the demand graph."""
    if H.number_of_nodes() == 0:
        return {}
    M = nx.algorithms.bipartite.maximum_matching(H, top_nodes=set(buyers))
    # networkx returns both directions; filter to buyer->seller only
    match: Dict[int, int] = {}
    for i in buyers:
        j = M.get(i)
        if j is not None and (i, j) in H.edges:
            match[i] = j
    return match

# Finds the constricted sets
def alternating_bfs_constricted_set(H: nx.Graph, buyers: List[int], match: Dict[int, int]) -> Tuple[Set[int], Set[int]]:
    """Compute (R, S) via alternating BFS from unmatched buyers.

    R: buyers reachable by alternating paths starting from unmatched buyers.
    S: sellers reachable from R via demand edges along alternating paths.
    """
    matched_buyers = set(match.keys())
    start = [i for i in buyers if i not in matched_buyers]
    R: Set[int] = set()
    S: Set[int] = set()

    from collections import deque
    dq = deque(start)
    R.update(start)

    while dq:
        u = dq.popleft()
        if u in buyers:  # from a buyer, traverse *non-matching* edges to sellers
            for v in H.neighbors(u):
                if v in S:
                    continue
                # do not restrict by matching here; next step will filter
                S.add(v)
                # from a seller, traverse only along its matched edge (if any)
                w = None
                # find the buyer matched to this seller
                for i, j in match.items():
                    if j == v:
                        w = i
                        break
                if w is not None and w not in R:
                    R.add(w)
                    dq.append(w)
        else:
            # shouldn't get here because we only push buyers into the queue
            pass
    return R, S

# Plots the graphs of each round
def plotting(H: nx.Graph, buyers: List[int], sellers: List[int], match: Dict[int, int], prices: Dict[int, float], title: str):
    if plt is None:
        return
    pos = {}
    # bipartite layout: buyers on left (x=0), sellers on right (x=1)
    y_b = list(range(len(sellers)))
    y_s = list(range(len(buyers)))
    for k, i in enumerate(sorted(sellers)):
        pos[i] = (0, -k)
    for k, j in enumerate(sorted(buyers)):
        pos[j] = (1, -k)

    plt.figure()
    nx.draw_networkx_nodes(H, pos, nodelist=sellers, node_shape='s')
    nx.draw_networkx_nodes(H, pos, nodelist=buyers, node_shape='o')
    nx.draw_networkx_edges(H, pos, alpha=0.5)

    # Highlight matching edges
    match_edges = [(i, j) for i, j in match.items()]
    nx.draw_networkx_edges(H, pos, edgelist=match_edges, width=2.5)

    # Node labels include prices for sellers
    labels = {i: str(i) for i in buyers}
    labels.update({j: f"{j}\n$p$={prices.get(j, 0):.2f}" for j in sellers})
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Solves the market by running through all rounds
def solve_market(mkt: Market, do_plot: bool, interactive: bool):
    round_no = 0
    last_match_size = -1
    max_rounds = 20

    while round_no < max_rounds:
        round_no += 1
        H, max_surplus = build_demand_graph(mkt)
        match = maximum_matching_on_demand(H, mkt.buyers)
        match_size = len(match)

        if interactive:
            print(f"\nRound {round_no}")
            print("Prices:", {j: round(p, 4) for j, p in mkt.prices.items()})
            print("Matching size:", match_size, "/", mkt.n)
            if match_size:
                print("Matching (buyer->seller):", dict(sorted(match.items())))
            plotting(H, mkt.buyers, mkt.sellers, match, mkt.prices, title=f"Round: {round_no} (|M|={match_size})")

        if match_size == mkt.n:
            if interactive:
                print("Perfect matching found. Terminating.")
            return match, mkt.prices, round_no

        # Compute constricted set (R buyers, S sellers)
        R, S = alternating_bfs_constricted_set(H, mkt.buyers, match)
        if interactive:
            print("Unmatched buyers:", [i for i in mkt.buyers if i not in match])
            print("Reachable buyers R:", sorted(R))
            print("Constricted sellers S:", sorted(S))

        # Update prices by 1 each round
        for j in S if S else mkt.sellers:
            mkt.prices[j] = mkt.prices.get(j, 0.0) + 1

    raise RuntimeError(f"Did not converge within {max_rounds} rounds. Consider checking the input graph.")

# Main function start
def main(argv: Optional[Sequence[str]] = None) -> int:
    # Load the Arguments
    p = argparse.ArgumentParser()
    p.add_argument("gml")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args(argv)

    # Checks to see if the gml file loaded, and loads the sellers and buyers into the mkt variable
    try:
        mkt = load_market(args.gml)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    G = mkt.G
    
    # Plots graph if flag is on
    if args.plot:
        # Create positions for bipartite layout
        pos = {}
        pos.update((n, (0, i)) for i, n in enumerate(mkt.sellers))  # sellers on left
        pos.update((n, (1, i)) for i, n in enumerate(mkt.buyers))  # buyers on right

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=mkt.sellers, node_color="skyblue", node_size=800, label="Sellers")
        nx.draw_networkx_nodes(G, pos, nodelist=mkt.buyers, node_color="lightgreen", node_size=800, label="Buyers")

        # Draw edges with valuation labels
        nx.draw_networkx_edges(G, pos)
        edge_labels = nx.get_edge_attributes(G, "valuation")
        nx.draw_networkx_edge_labels(G, pos, label_pos=0.75, edge_labels=edge_labels, font_size=8)

        # Draw node labels
        nx.draw_networkx_labels(G, pos)

        # Title and layout
        plt.title("Bipartite Market Graph (Sellers vs Buyers)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Solves the market round by round, also shows it if the interactive flag is checked
    try:
        match, prices, rounds = solve_market(mkt, do_plot=args.plot, interactive=args.interactive)
    except Exception as e:
        print(f"Error during solving: {e}", file=sys.stderr)
        return 3

    # Final report
    print("\n=== Market cleared ===")
    print(f"Rounds: {rounds}")
    print("Final prices:", {j: round(p, 4) for j, p in sorted(prices.items())})
    print("Final matching (buyer->seller):", dict(sorted(match.items())))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
