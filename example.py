# example_stock.py
# usage of DynamicBayesianNetwork for stock price direction
from typing import Dict, Any, Tuple
from dbn import DynamicBayesianNetwork 


def build_stock_dbn() -> DynamicBayesianNetwork:
    """
    Simple stock DBN:
      - MarketSentiment_t: Bullish / Bearish
      - PriceMove_t: Increase / Decrease
      Dependencies:
        - PriceMove_{t-1} -> PriceMove_t   (momentum)
        - MarketSentiment_t -> PriceMove_t (macro mood)
    """
    dbn = DynamicBayesianNetwork("Stock-DBN")

    # nodes
    dbn.add_node("MarketSentiment")
    dbn.add_node("PriceMove")

    # edges
    dbn.add_inter_edge("PriceMove", "PriceMove")         # PriceMove_{t-1} -> PriceMove_t
    dbn.add_intra_edge("MarketSentiment", "PriceMove")   # MarketSentiment_t -> PriceMove_t

    # CPTs
    # 1) MarketSentiment_t: no parents
    #    maybe the market is bullish 60% of the time
    
    market_cpt = {
        (): {"Bullish": 0.6, "Bearish": 0.4}
    }
    dbn.set_cpt("MarketSentiment", market_cpt)

    # 2) PriceMove_t | (MarketSentiment_t, PriceMove_{t-1})
    price_cpt = {
        # if market is bullish and it increased yesterday, high chance to increase again
        ("Bullish", "Increase"): {"Increase": 0.8, "Decrease": 0.2},
        ("Bearish", "Increase"): {"Increase": 0.55, "Decrease": 0.45},
        # if market is bullish but it decreased yesterday, could rebound
        ("Bullish", "Decrease"): {"Increase": 0.6, "Decrease": 0.4},
        ("Bearish", "Decrease"): {"Increase": 0.3, "Decrease": 0.7},
    }
    dbn.set_cpt("PriceMove", price_cpt)

    return dbn


def main():
    dbn = build_stock_dbn()

    # Evidence at t=0 
    # Example: today market was Bullish and price Increased
    evidence: Dict[Tuple[str, int], Any] = {
        ("MarketSentiment", 0): "Bullish",
        ("PriceMove", 0): "Increase",
    }

    market_t1_dist = dbn.infer_node("MarketSentiment", t=1, evidence={})
    most_likely_market_t1 = max(market_t1_dist, key=market_t1_dist.get)

    evidence[("MarketSentiment", 1)] = most_likely_market_t1

    # infer PriceMove at t=1
    price_t1_dist = dbn.infer_node("PriceMove", t=1, evidence=evidence)

    print("Observed at t=0:")
    print("  MarketSentiment_0 =", evidence[("MarketSentiment", 0)])
    print("  PriceMove_0       =", evidence[("PriceMove", 0)])
    print()

    print("Predicted MarketSentiment_1 (from root CPT):", market_t1_dist)
    print(f"Using MarketSentiment_1 = {most_likely_market_t1}")
    print()

    print("P(PriceMove_1 | evidence) =")
    for v, p in price_t1_dist.items():
        print(f"  {v:9s}: {p:.3f}")

    predicted_label = max(price_t1_dist, key=price_t1_dist.get)
    print("\n=> Predicted next move:", predicted_label)


if __name__ == "__main__":
    main()