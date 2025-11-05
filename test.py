
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import networkx as nx
except ImportError as e:
    print("This script requires networkx. Install with `pip install networkx matplotlib`.", file=sys.stderr)
    raise

# Matplotlib is optional unless --plot is passed
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


ValKeyCandidates: Sequence[str] = ("valuation", "value", "weight")


@dataclass
class Market:
    G: nx.Graph
    sellers: List[int]
    buyers: List[int]
    n: int
    prices: Dict[int, float]
    valkey: str


# ------------------------------
# Loading and validation helpers
# ------------------------------

def load_market(path: str) -> Market:
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        G = nx.read_gml(path, label='id')
    except Exception as e:
        raise ValueError(f"Failed to read GML: {e}")

    # Ensure undirected for our use; valuations are read from edges regardless of orientation
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G.to_undirected())

    # Node ids should be integers 0..2n-1
    try:
        node_ids = sorted(int(u) for u in G.nodes())
    except Exception:
        raise ValueError("Node identifiers must be integers 0..2n-1 in the GML file.")
    if not node_ids:
        raise ValueError("Graph has no nodes.")
    if node_ids[0] != 0:
        raise ValueError("Smallest node id must be 0.")
    max_id = node_ids[-1]
    m = max_id + 1
    if m % 2 != 0:
        raise ValueError("Number of nodes must be even (2n).")
    n = m // 2
    expected_ids = list(range(0, 2 * n))
    if node_ids != expected_ids:
        raise ValueError("Node ids must be exactly 0..(2n-1) with no gaps.")

    sellers = list(range(0, n))
    buyers = list(range(n, 2 * n))

    # Determine valuation key by inspecting the first edge that has a matching key
    found_key: Optional[str] = None
    for (u, v, data) in G.edges(data=True):
        for k in ValKeyCandidates:
            if k in data:
                found_key = k
                break
        if found_key:
            break
    if not found_key:
        raise ValueError(
            f"Edges must carry a valuation attribute (one of {list(ValKeyCandidates)})."
        )

    # Initialize prices from seller node attributes, default 0.0
    prices: Dict[int, float] = {}
    for a in sellers:
        attr = G.nodes[a]
        p = float(attr.get("price", 0.0))
        if math.isnan(p) or math.isinf(p) or p < 0:
            raise ValueError(f"Invalid price on seller node {a}: {p}")
        prices[a] = p

    return Market(G=G, sellers=sellers, buyers=buyers, n=n, prices=prices, valkey=found_key)


# ------------------------------
# Core market-clearing procedures
# ------------------------------

def valuation(mkt: Market, buyer: int, seller: int) -> float:
    """Return valuation v_{buyer, seller}. If no edge, treat as -inf (unavailable)."""
    data = mkt.G.get_edge_data(buyer, seller, default=None)
    if data is None:
        return float("-inf")
    val = data.get(mkt.valkey, None)
    if val is None:
        return float("-inf")
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Non-numeric valuation on edge ({buyer}, {seller}).")


def build_demand_graph(mkt: Market) -> Tuple[nx.Graph, Dict[int, float]]:
    """Construct preference (demand) graph at current prices.

    For each buyer i in B, compute s_i(j) = v(i,j) - p_j. Keep edges to sellers achieving
    the maximum nonnegative surplus. Return the demand graph and a dict of each buyer's
    max surplus value.
    """
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


def compute_epsilon(mkt: Market, R: Set[int], S: Set[int]) -> float:
    """Smallest price increase ε on sellers in S so that some buyer in R becomes
    indifferent with a seller outside S (causing demand to expand) or an edge appears.

    ε = min_{i in R} [ max_{j in S} (v_ij - p_j) - max_{k not in S} (v_ik - p_k) ]_+.
    If a buyer has no seller outside S, we ignore them (treated as +inf).
    """
    eps = float("inf")
    for i in R:
        best_in_S = float("-inf")
        best_out_S = float("-inf")
        for j in mkt.sellers:
            vij = valuation(mkt, i, j)
            if vij == float("-inf"):
                continue
            s = vij - mkt.prices[j]
            if j in S:
                if s > best_in_S:
                    best_in_S = s
            else:
                if s > best_out_S:
                    best_out_S = s
        if best_out_S == float("-inf"):
            # no outside option; ignore this i
            continue
        gap = best_in_S - best_out_S
        if gap > 1e-12 and gap < eps:
            eps = gap
    if eps == float("inf") or eps <= 0:
        # fallback: tiny positive step to avoid stalling if numerics are weird
        eps = 1e-6
    return eps


# ------------------------------
# Plotting utilities
# ------------------------------

def maybe_draw(H: nx.Graph, buyers: List[int], sellers: List[int], match: Dict[int, int], prices: Dict[int, float], title: str):
    if plt is None:
        return
    pos = {}
    # bipartite layout: buyers on left (x=0), sellers on right (x=1)
    y_b = list(range(len(buyers)))
    y_s = list(range(len(sellers)))
    for k, i in enumerate(sorted(buyers)):
        pos[i] = (0, -k)
    for k, j in enumerate(sorted(sellers)):
        pos[j] = (1, -k)

    plt.figure()
    nx.draw_networkx_nodes(H, pos, nodelist=buyers, node_shape='s')
    nx.draw_networkx_nodes(H, pos, nodelist=sellers, node_shape='o')
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
    plt.show(block=False)
    plt.pause(0.5)


# ------------------------------
# Main iterative solver
# ------------------------------

def solve_market(mkt: Market, do_plot: bool, interactive: bool, max_rounds: int = 5000) -> Tuple[Dict[int, int], Dict[int, float], int]:
    round_no = 0
    last_match_size = -1

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

        if do_plot:
            maybe_draw(H, mkt.buyers, mkt.sellers, match, mkt.prices, title=f"Demand graph — round {round_no} (|M|={match_size})")

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

        # If S is empty (can happen if unmatched buyers have no edges), try to add tiny epsilon to all sellers
        if not S:
            eps = 1e-6
        else:
            eps = compute_epsilon(mkt, R, S)
        if interactive:
            print(f"Epsilon price increase: {eps:.6g}")

        # Update prices
        for j in S if S else mkt.sellers:
            mkt.prices[j] = mkt.prices.get(j, 0.0) + eps

        # Convergence guard: if nothing changes in matching size for too long
        if match_size == last_match_size:
            pass
        last_match_size = match_size

    raise RuntimeError(f"Did not converge within {max_rounds} rounds. Consider checking the input graph.")


# ------------------------------
# CLI
# ------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Market-clearing via ascending auction on a bipartite graph (GML)")
    p.add_argument("gml", help="Path to market.gml (Graph Modelling Language)")
    p.add_argument("--plot", action="store_true", help="Plot the demand graph each round")
    p.add_argument("--interactive", action="store_true", help="Verbose round-by-round output")
    p.add_argument("--max-rounds", type=int, default=5000, help="Iteration cap (default: 5000)")
    p.add_argument("--debug", action="store_true", help="Show full traceback on errors")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import traceback
    args = parse_args(argv)
    try:
        mkt = load_market(args.gml)
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        match, prices, rounds = solve_market(mkt, do_plot=args.plot, interactive=args.interactive, max_rounds=args.max_rounds)
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        print(f"Error during solving: {e}", file=sys.stderr)
        return 3

    # Final report
    print("\n=== Market cleared ===")
    print(f"Rounds: {rounds}")
    print("Final prices:", {j: round(p, 4) for j, p in sorted(prices.items())})
    print("Final matching (buyer->seller):", dict(sorted(match.items())))

    # Compute total surplus at final prices for reporting
    total_value = 0.0
    for i, j in match.items():
        v = valuation(mkt, i, j)
        total_value += v
    revenue = sum(prices.values())
    print(f"Total value of matched pairs: {total_value:.4f}")
    print(f"Sum of seller prices: {revenue:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
