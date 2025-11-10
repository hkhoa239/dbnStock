from typing import Dict, List, Tuple, Any


class DynamicBayesianNetwork:
    """
    A lightweight Dynamic Bayesian Network (DBN) for temporal modeling.
    
    Supports:
        - add_node: Add nodes to the network
        - add_intra_edge: Add edges within a time slice (X_t -> Y_t)
        - add_inter_edge: Add edges across time slices (X_{t-1} -> Y_t)
        - set_cpt: Set Conditional Probability Tables for nodes
        - unroll: Unroll the network for T time steps
        - infer_node: Perform simple forward inference given evidence
        - update_cpt_from_data: Adaptive learning from streaming data
    """
    
    def __init__(self, name: str = "DBN"):
        """
        Initialize the Dynamic Bayesian Network.
        
        Args:
            name: Name identifier for the network
        """
        self.name = name
        self.slice_nodes: List[str] = []
        self.intra_edges: List[Tuple[str, str]] = []
        self.inter_edges: List[Tuple[str, str]] = []
        self.cpt: Dict[str, Dict[Tuple, Dict[Any, float]]] = {}

    def add_node(self, node: str) -> None:
        """
        Add a node to the network time slice.
        
        Args:
            node: Node name to add
        """
        if node not in self.slice_nodes:
            self.slice_nodes.append(node)

    def add_intra_edge(self, parent: str, child: str) -> None:
        """
        Add an edge within a time slice (parent_t -> child_t).
        
        Args:
            parent: Parent node name
            child: Child node name
        """
        self.intra_edges.append((parent, child))

    def add_inter_edge(self, parent_prev: str, child_curr: str) -> None:
        """
        Add an edge across time slices (parent_{t-1} -> child_t).
        
        Args:
            parent_prev: Parent node name at time t-1
            child_curr: Child node name at time t
        """
        self.inter_edges.append((parent_prev, child_curr))

    def set_cpt(self, node: str, cpt_table: Dict[Tuple, Dict[Any, float]]) -> None:
        """
        Set the Conditional Probability Table (CPT) for a node.
        
        Args:
            node: Node name
            cpt_table: Dictionary mapping parent values to probability distributions
                      Format: {parent_values_tuple: {node_value: probability}}
                      
        Example:
            For a Decision node with parents (Price, Volatility):
            {
                ('Up', 'LowVol'): {'Buy': 0.8, 'Hold': 0.2},
                ('Down', 'HighVol'): {'Sell': 0.7, 'Hold': 0.3}
            }
            
            For a root node with no parents:
            {(): {'High': 0.6, 'Low': 0.4}}
        """
        self.cpt[node] = cpt_table

    def get_parents(self, node: str) -> List[Tuple[str, int]]:
        """
        Get all parent nodes for a given node.
        
        Args:
            node: Node name
            
        Returns:
            List of tuples (parent_node_name, time_offset)
            where time_offset is 0 for intra-slice edges and -1 for inter-slice edges
        """
        parents = []
        
        # Add intra-slice parents (same time step)
        for parent, child in self.intra_edges:
            if child == node:
                parents.append((parent, 0))
        
        # Add inter-slice parents (previous time step)
        for parent, child in self.inter_edges:
            if child == node:
                parents.append((parent, -1))
        
        return parents

    def infer_node(
        self,
        node: str,
        t: int,
        evidence: Dict[Tuple[str, int], Any],
    ) -> Dict[Any, float]:
        """
        Perform simple forward inference for a node at time t.
        
        Args:
            node: Node name to infer
            t: Time step
            evidence: Dictionary mapping (node_name, time) to observed values
            
        Returns:
            Probability distribution over node values: {value: probability}
            
        Raises:
            ValueError: If CPT is not defined or required evidence is missing
        """
        # Get node's CPT
        if node not in self.cpt:
            raise ValueError(f"No CPT defined for node '{node}'")
        node_cpt = self.cpt[node]
        
        # Get parent nodes and their values from evidence
        parents = self.get_parents(node)
        
        # Handle root nodes (no parents)
        if len(parents) == 0:
            return node_cpt[()]
        
        # Gather parent values from evidence
        parent_vals = []
        for parent_node, time_offset in parents:
            evidence_key = (parent_node, t + time_offset)
            if evidence_key not in evidence:
                raise ValueError(
                    f"Missing evidence for parent '{parent_node}' "
                    f"at time {t + time_offset}"
                )
            parent_vals.append(evidence[evidence_key])
        
        # Look up probability distribution in CPT
        parent_vals_tuple = tuple(parent_vals)
        if parent_vals_tuple not in node_cpt:
            raise ValueError(
                f"No CPT entry for node '{node}' with parent values {parent_vals_tuple}"
            )
        
        return node_cpt[parent_vals_tuple]

    def unroll(self, T: int) -> List[List[Tuple[str, int]]]:
        """
        Unroll the network structure over T time steps.
        
        This creates a conceptual representation of the temporal network
        without building an explicit large graph object.
        
        Args:
            T: Number of time steps to unroll
            
        Returns:
            List of time slices, where each slice contains (node_name, time) tuples
        """
        slices = []
        for t in range(T):
            slices.append([(node, t) for node in self.slice_nodes])
        return slices

    def update_cpt_from_data(
        self,
        node: str,
        parent_values: Tuple,
        observed_value: Any,
        lr: float = 0.1
    ) -> None:
        """
        Adaptively update CPT based on observed data (online learning).
        
        Uses exponential moving average to shift probability mass toward 
        the observed value, enabling the network to adapt to streaming data.
        
        Args:
            node: Node name to update
            parent_values: Tuple of parent values for this observation
            observed_value: The observed value for the node
            lr: Learning rate (0 < lr <= 1). Higher values adapt faster
            
        Raises:
            ValueError: If the node has no CPT defined
        """
        if node not in self.cpt:
            raise ValueError(f"CPT for node '{node}' not found")
        
        # Initialize parent_values entry if it doesn't exist
        if parent_values not in self.cpt[node]:
            self.cpt[node][parent_values] = {}
        
        distribution = self.cpt[node][parent_values]
        
        # Initialize observed_value with small probability if not present
        if observed_value not in distribution:
            distribution[observed_value] = 1e-6
        
        # Update probabilities using exponential moving average
        all_values = list(distribution.keys())
        for value in all_values:
            if value == observed_value:
                # Increase probability of observed value
                distribution[value] = distribution[value] * (1 - lr) + lr
            else:
                # Decrease probability of other values
                distribution[value] = distribution[value] * (1 - lr)
        
        # Normalize to ensure probabilities sum to 1
        total = sum(distribution.values())
        if total > 0:
            for value in all_values:
                distribution[value] /= total