"""
Visualization module for Dynamic Bayesian Networks.
Displays network structure with nodes, intra-slice, and inter-slice edges.

Requirements:
    pip install matplotlib networkx

Or install from requirements.txt:
    pip install -r requirements.txt
"""
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    IMPORT_ERROR = str(e)

from typing import Optional, Tuple
from dbn import DynamicBayesianNetwork


class DBNVisualizer:
    """Visualizer for Dynamic Bayesian Network structures."""
    
    def __init__(self, dbn: DynamicBayesianNetwork):
        """
        Initialize the visualizer.
        
        Args:
            dbn: DynamicBayesianNetwork instance to visualize
        """
        self.dbn = dbn
        
    def visualize_structure(
        self,
        time_slices: int = 2,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize the DBN structure across multiple time slices.
        
        Args:
            time_slices: Number of time slices to display (default: 2)
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
            show: Whether to display the plot (default: True)
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                f"Visualization libraries not available: {IMPORT_ERROR}\n"
                "Please install required packages:\n"
                "  pip install matplotlib networkx\n"
                "Or:\n"
                "  pip install -r requirements.txt"
            )
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes for each time slice
        node_positions = {}
        y_spacing = 2.0
        x_spacing = 4.0
        
        # Calculate vertical positions for nodes
        num_nodes = len(self.dbn.slice_nodes)
        y_positions = {}
        for i, node in enumerate(self.dbn.slice_nodes):
            y_positions[node] = (num_nodes - 1 - i) * y_spacing
        
        # Add nodes and edges for each time slice
        for t in range(time_slices):
            x_pos = t * x_spacing
            
            # Add nodes for this time slice
            for node in self.dbn.slice_nodes:
                node_label = f"{node}_{{{t}}}"
                G.add_node(node_label)
                node_positions[node_label] = (x_pos, y_positions[node])
            
            # Add intra-slice edges (within time t)
            for parent, child in self.dbn.intra_edges:
                parent_label = f"{parent}_{{{t}}}"
                child_label = f"{child}_{{{t}}}"
                G.add_edge(parent_label, child_label, edge_type='intra')
            
            # Add inter-slice edges (from t-1 to t)
            if t > 0:
                for parent, child in self.dbn.inter_edges:
                    parent_label = f"{parent}_{{{t-1}}}"
                    child_label = f"{child}_{{{t}}}"
                    G.add_edge(parent_label, child_label, edge_type='inter')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, node_positions, 
            node_color='lightblue',
            node_size=3000,
            alpha=0.9,
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, node_positions,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # Separate edges by type for different styling
        intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'intra']
        inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'inter']
        
        # Draw intra-slice edges (solid lines)
        nx.draw_networkx_edges(
            G, node_positions,
            edgelist=intra_edges,
            edge_color='black',
            style='solid',
            width=2,
            alpha=0.7,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw inter-slice edges (dashed lines)
        nx.draw_networkx_edges(
            G, node_positions,
            edgelist=inter_edges,
            edge_color='red',
            style='dashed',
            width=2,
            alpha=0.7,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Add title and legend
        plt.title(f"Dynamic Bayesian Network: {self.dbn.name}", 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2, label='Intra-slice edge (same time)'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Inter-slice edge (temporal)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add time slice labels
        for t in range(time_slices):
            x_pos = t * x_spacing
            ax.text(x_pos, max(y_positions.values()) + 1, f't = {t}',
                   fontsize=14, fontweight='bold', ha='center')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def print_network_info(self) -> None:
        """Print detailed information about the network structure."""
        print("=" * 60)
        print(f"Dynamic Bayesian Network: {self.dbn.name}")
        print("=" * 60)
        
        print("\n📊 NODES:")
        for i, node in enumerate(self.dbn.slice_nodes, 1):
            parents = self.dbn.get_parents(node)
            if parents:
                parent_str = ", ".join([f"{p}(t{offset:+d})" for p, offset in parents])
                print(f"  {i}. {node} | Parents: {parent_str}")
            else:
                print(f"  {i}. {node} | Root node (no parents)")
        
        print("\n🔗 INTRA-SLICE EDGES (within time slice):")
        if self.dbn.intra_edges:
            for i, (parent, child) in enumerate(self.dbn.intra_edges, 1):
                print(f"  {i}. {parent}_t → {child}_t")
        else:
            print("  None")
        
        print("\n⏰ INTER-SLICE EDGES (across time slices):")
        if self.dbn.inter_edges:
            for i, (parent, child) in enumerate(self.dbn.inter_edges, 1):
                print(f"  {i}. {parent}_{{t-1}} → {child}_t")
        else:
            print("  None")
        
        print("\n📋 CONDITIONAL PROBABILITY TABLES:")
        for node in self.dbn.slice_nodes:
            if node in self.dbn.cpt:
                print(f"\n  {node}:")
                cpt = self.dbn.cpt[node]
                for parent_vals, dist in cpt.items():
                    if parent_vals == ():
                        print(f"    P({node}) =")
                    else:
                        parents = self.dbn.get_parents(node)
                        parent_names = [f"{p}(t{offset:+d})" for p, offset in parents]
                        print(f"    P({node} | {', '.join(parent_names)} = {parent_vals}) =")
                    for value, prob in dist.items():
                        print(f"      {value}: {prob:.3f}")
        
        print("\n" + "=" * 60)


def visualize_dbn(
    dbn: DynamicBayesianNetwork,
    time_slices: int = 2,
    show_info: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Convenience function to visualize a DBN.
    
    Args:
        dbn: DynamicBayesianNetwork instance
        time_slices: Number of time slices to display
        show_info: Whether to print network information
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    visualizer = DBNVisualizer(dbn)
    
    if show_info:
        visualizer.print_network_info()
    
    visualizer.visualize_structure(
        time_slices=time_slices,
        save_path=save_path,
        show=show
    )


if __name__ == "__main__":
    # Example usage
    from example import build_stock_dbn
    
    print("Loading Stock DBN example...\n")
    dbn = build_stock_dbn()
    
    # Visualize the network
    visualize_dbn(
        dbn,
        time_slices=3,
        show_info=True,
        save_path="stock_dbn_structure.png",
        show=False  # Set to True if you want to display interactively
    )
    
    print("\nVisualization complete!")

