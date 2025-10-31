"""
Visualization framework for Cognitive Load Traces.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CLTVisualizer:
    """
    Visualization tools for CLT analysis including temporal curves and simplex diagrams.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style name
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_temporal_traces(
        self,
        clt_trace: np.ndarray,
        cli_trace: np.ndarray,
        error_events: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot temporal load curves showing IL, EL, GL, and CLI over time.
        
        Args:
            clt_trace: [T, 3] array of (IL, EL, GL)
            cli_trace: [T] array of composite load indices
            error_events: Optional list of error step indices to highlight
            save_path: Optional path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        steps = np.arange(len(clt_trace))
        
        # Plot IL, EL, GL
        ax.plot(steps, clt_trace[:, 0], label='IL (Intrinsic)', linewidth=2, alpha=0.8)
        ax.plot(steps, clt_trace[:, 1], label='EL (Extraneous)', linewidth=2, alpha=0.8)
        ax.plot(steps, clt_trace[:, 2], label='GL (Germane)', linewidth=2, alpha=0.8)
        
        # Plot CLI
        ax.plot(steps, cli_trace, label='CLI (Composite)', linewidth=3, 
                linestyle='--', color='black', alpha=0.7)
        
        # Highlight error events
        if error_events:
            for error_step in error_events:
                ax.axvline(x=error_step, color='red', linestyle=':', 
                          alpha=0.5, linewidth=2)
                ax.plot(error_step, cli_trace[error_step], 'ro', 
                       markersize=10, label='Error' if error_step == error_events[0] else '')
        
        ax.set_xlabel('Decoding Step', fontsize=12)
        ax.set_ylabel('Load Value', fontsize=12)
        ax.set_title('Cognitive Load Traces Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_simplex(
        self,
        clt_trace: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ):
        """
        Plot IL-EL-GL simplex showing cognitive strategy space.
        
        Args:
            clt_trace: [T, 3] array of (IL, EL, GL)
            labels: Optional list of labels for each point
            save_path: Optional path to save figure
            figsize: Figure size
        """
        # Convert to simplex coordinates (barycentric)
        # Normalize to ensure points lie on simplex
        normalized_clt = clt_trace / (clt_trace.sum(axis=1, keepdims=True) + 1e-10)
        
        # Convert to 2D using:
        # x = (2 * IL + GL) / (2 * sum)
        # y = (sqrt(3) * GL) / (2 * sum)
        x = (2 * normalized_clt[:, 0] + normalized_clt[:, 2]) / 2
        y = (np.sqrt(3) * normalized_clt[:, 2]) / 2
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw simplex triangle
        vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        triangle = mpatches.Polygon(vertices, fill=False, edgecolor='black', 
                                   linewidth=2)
        ax.add_patch(triangle)
        
        # Plot points
        scatter = ax.scatter(x, y, c=range(len(clt_trace)), 
                           cmap='viridis', s=50, alpha=0.6)
        
        # Add vertex labels
        ax.text(-0.05, -0.05, 'IL', fontsize=14, fontweight='bold')
        ax.text(0.95, -0.05, 'EL', fontsize=14, fontweight='bold')
        ax.text(0.45, np.sqrt(3)/2 + 0.05, 'GL', fontsize=14, fontweight='bold')
        
        # Draw grid lines
        for i in range(5):
            # Lines parallel to IL-EL
            alpha = i / 4
            p1 = (1-alpha) * vertices[0] + alpha * vertices[2]
            p2 = (1-alpha) * vertices[1] + alpha * vertices[2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.2, linewidth=0.5)
            
            # Lines parallel to IL-GL
            p1 = (1-alpha) * vertices[0] + alpha * vertices[1]
            p2 = (1-alpha) * vertices[2] + alpha * vertices[1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.2, linewidth=0.5)
            
            # Lines parallel to EL-GL
            p1 = (1-alpha) * vertices[1] + alpha * vertices[0]
            p2 = (1-alpha) * vertices[2] + alpha * vertices[0]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.2, linewidth=0.5)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal')
        ax.set_title('Cognitive Load Simplex (IL-EL-GL Space)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Decoding Step')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_layer_time_heatmap(
        self,
        layer_loads: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot heatmap of load across layers and time.
        
        Args:
            layer_loads: [num_layers, T] array of load values
            save_path: Optional path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(layer_loads, cmap='YlOrRd', cbar_kws={'label': 'Load Value'},
                   ax=ax, xticklabels=50)
        
        ax.set_xlabel('Decoding Step', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title('Cognitive Load Distribution Across Layers', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_radar_profile(
        self,
        load_profile: Dict[str, float],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ):
        """
        Plot radar/spider chart of overall cognitive characteristics.
        
        Args:
            load_profile: Dictionary of metric -> value
            save_path: Optional path to save figure
            figsize: Figure size
        """
        categories = list(load_profile.keys())
        values = list(load_profile.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first value to end to close the plot
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Cognitive Load Profile', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_correlation_analysis(
        self,
        clt_trace: np.ndarray,
        cli_trace: np.ndarray,
        error_events: List[int],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Plot correlation analysis showing relationship between CLT and errors.
        
        Args:
            clt_trace: [T, 3] array of (IL, EL, GL)
            cli_trace: [T] array of composite load indices
            error_events: List of error step indices
            save_path: Optional path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Create binary error indicator
        error_indicator = np.zeros(len(cli_trace))
        for error in error_events:
            error_indicator[error] = 1
        
        # Plot CLI distribution for error vs non-error steps
        axes[0].hist(cli_trace[error_indicator == 0], bins=30, alpha=0.5, 
                    label='No Error', density=True)
        axes[0].hist(cli_trace[error_indicator == 1], bins=30, alpha=0.5, 
                    label='Error', density=True, color='red')
        axes[0].set_xlabel('CLI Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('CLI Distribution')
        axes[0].legend()
        
        # Plot correlation by component
        components = ['IL', 'EL', 'GL']
        correlations = []
        for i, comp in enumerate(components):
            corr = np.corrcoef(clt_trace[:, i], error_indicator)[0, 1]
            correlations.append(corr)
        
        axes[1].bar(components, correlations, color=['blue', 'orange', 'green'])
        axes[1].set_ylabel('Correlation with Error')
        axes[1].set_title('Component-Error Correlation')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Plot CLI time series with error annotations
        axes[2].plot(cli_trace, alpha=0.7, label='CLI')
        axes[2].scatter(error_events, cli_trace[error_events], 
                       color='red', s=100, zorder=5, label='Error')
        axes[2].axhline(y=np.percentile(cli_trace, 75), linestyle='--', 
                       color='orange', label='75th percentile')
        axes[2].set_xlabel('Decoding Step')
        axes[2].set_ylabel('CLI Value')
        axes[2].set_title('CLI vs Error Events')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def create_interactive_trace(
        self,
        clt_trace: np.ndarray,
        cli_trace: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Create interactive Plotly visualization of CLT traces.
        
        Args:
            clt_trace: [T, 3] array of (IL, EL, GL)
            cli_trace: [T] array of composite load indices
            save_path: Optional path to save HTML
        """
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('CLT Components', 'Composite Load Index'))
        
        steps = np.arange(len(clt_trace))
        
        # Add CLT components
        fig.add_trace(
            go.Scatter(x=steps, y=clt_trace[:, 0], name='IL', mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=clt_trace[:, 1], name='EL', mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=clt_trace[:, 2], name='GL', mode='lines'),
            row=1, col=1
        )
        
        # Add CLI
        fig.add_trace(
            go.Scatter(x=steps, y=cli_trace, name='CLI', mode='lines', 
                      line=dict(color='black', dash='dash', width=3)),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Cognitive Load Traces (Interactive)")
        fig.update_xaxes(title_text="Decoding Step", row=2, col=1)
        fig.update_yaxes(title_text="Load Value", row=1, col=1)
        fig.update_yaxes(title_text="CLI", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

