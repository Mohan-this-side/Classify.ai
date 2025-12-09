"""
Reasoning Plot Generator

Generates comprehensive visualizations showing problem detection → reasoning → solution → results.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class ReasoningPlotGenerator:
    """Generate reasoning visualizations showing agent problem-solving workflow"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.ReasoningPlotGenerator")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_all_plots(self, reasoning_data: Dict[str, Any]) -> List[str]:
        """Generate all reasoning plots"""
        generated_plots = []
        
        try:
            # 1. Problem Detection Summary
            if "all_problems" in reasoning_data:
                plot_path = self.generate_problem_detection_summary(reasoning_data["all_problems"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 2. Missing Data Handling Flow
            if "cleaning" in reasoning_data:
                plot_path = self.generate_missing_data_handling(reasoning_data["cleaning"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 3. Class Imbalance Resolution
            if "imbalance" in reasoning_data:
                plot_path = self.generate_imbalance_resolution(reasoning_data["imbalance"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 4. Multicollinearity Removal Network
            if "multicollinearity" in reasoning_data:
                plot_path = self.generate_multicollinearity_removal(reasoning_data["multicollinearity"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 5. Feature Engineering Summary
            if "feature_engineering" in reasoning_data:
                plot_path = self.generate_feature_engineering_summary(reasoning_data["feature_engineering"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 6. Temporal Split Visualization
            if "temporal" in reasoning_data:
                plot_path = self.generate_temporal_split(reasoning_data["temporal"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 7. Location Missing Pattern Heatmap
            if "location_patterns" in reasoning_data:
                plot_path = self.generate_location_patterns(reasoning_data["location_patterns"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 8. Outlier Treatment
            if "cleaning" in reasoning_data:
                plot_path = self.generate_outlier_treatment(reasoning_data["cleaning"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 9. Model Selection Flowchart
            if "model_selection" in reasoning_data:
                plot_path = self.generate_model_selection_flowchart(reasoning_data["model_selection"])
                if plot_path:
                    generated_plots.append(plot_path)
            
            # 10. Comprehensive Problem-Solution Matrix
            plot_path = self.generate_problem_solution_matrix(reasoning_data)
            if plot_path:
                generated_plots.append(plot_path)
            
            self.logger.info(f"✅ Generated {len(generated_plots)} reasoning plots")
            return generated_plots
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning plots: {e}", exc_info=True)
            return generated_plots
    
    def generate_problem_detection_summary(self, problems: Dict[str, Any]) -> Optional[str]:
        """Generate bar chart showing all problems detected with severity"""
        try:
            problem_names = []
            severities = []
            statuses = []
            
            problem_mapping = {
                "class_imbalance": "Class Imbalance",
                "missing_data": "Missing Data",
                "multicollinearity": "Multicollinearity",
                "data_leakage": "Data Leakage",
                "temporal_patterns": "Temporal Patterns",
                "location_missing_patterns": "Location Patterns",
                "weak_feature_correlations": "Weak Correlations",
                "outliers": "Outliers",
                "target_missing": "Target Missing",
                "high_cardinality": "High Cardinality"
            }
            
            for key, name in problem_mapping.items():
                if key in problems and problems[key].get("detected", False):
                    problem_names.append(name)
                    severity = problems[key].get("severity", "medium")
                    severities.append(severity)
                    statuses.append("Detected")
            
            if not problem_names:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#3498db', 'unknown': '#95a5a6'}
            bar_colors = [colors.get(s, '#95a5a6') for s in severities]
            
            bars = ax.barh(problem_names, [1] * len(problem_names), color=bar_colors, alpha=0.7)
            ax.set_xlabel('Status', fontsize=12, fontweight='bold')
            ax.set_title('Critical Problems Detected in Dataset', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlim(0, 1.2)
            
            # Add severity labels
            for i, (bar, severity) in enumerate(zip(bars, severities)):
                ax.text(0.5, bar.get_y() + bar.get_height()/2, 
                       severity.upper(), ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
            
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plot_path = self.output_dir / "problem_detection_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating problem detection summary: {e}")
            return None
    
    def generate_missing_data_handling(self, cleaning_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate bar chart showing missing data handling: Column → Missing → Action → Result"""
        try:
            columns = []
            missing_before = []
            missing_after = []
            strategies = []
            
            for col, col_reasoning in cleaning_reasoning.items():
                if isinstance(col_reasoning, dict) and "problem" in col_reasoning:
                    problem = col_reasoning["problem"]
                    if problem.get("type") == "missing_values":
                        columns.append(col[:20])  # Truncate long names
                        missing_before.append(problem.get("count", 0))
                        missing_after.append(col_reasoning.get("result", {}).get("missing_after", 0))
                        strategies.append(col_reasoning.get("action", {}).get("strategy", "unknown"))
            
            if not columns:
                return None
            
            # Limit to top 15 columns
            if len(columns) > 15:
                indices = np.argsort(missing_before)[-15:][::-1]
                columns = [columns[i] for i in indices]
                missing_before = [missing_before[i] for i in indices]
                missing_after = [missing_after[i] for i in indices]
                strategies = [strategies[i] for i in indices]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Before/After comparison
            x_pos = np.arange(len(columns))
            width = 0.35
            
            bars1 = ax1.bar(x_pos - width/2, missing_before, width, label='Before', color='#e74c3c', alpha=0.7)
            bars2 = ax1.bar(x_pos + width/2, missing_after, width, label='After', color='#2ecc71', alpha=0.7)
            
            ax1.set_xlabel('Column', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Missing Values', fontsize=11, fontweight='bold')
            ax1.set_title('Missing Values: Before → After', fontsize=12, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(columns, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Strategy distribution
            strategy_counts = pd.Series(strategies).value_counts()
            ax2.bar(strategy_counts.index, strategy_counts.values, color='#3498db', alpha=0.7)
            ax2.set_xlabel('Imputation Strategy', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Number of Columns', fontsize=11, fontweight='bold')
            ax2.set_title('Imputation Strategies Used', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / "missing_data_handling.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating missing data handling plot: {e}")
            return None
    
    def generate_imbalance_resolution(self, imbalance_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate before/after class distribution + SMOTE application"""
        try:
            detection = imbalance_reasoning.get("detection", {})
            handling = imbalance_reasoning.get("handling", {})
            
            if not detection:
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Before distribution
            if "distribution" in detection:
                dist_before = detection["distribution"]
                classes = list(dist_before.keys())
                counts_before = list(dist_before.values())
                
                axes[0].bar(classes, counts_before, color=['#3498db', '#e74c3c'], alpha=0.7)
                axes[0].set_xlabel('Class', fontsize=11, fontweight='bold')
                axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
                axes[0].set_title(f'Before: Imbalance Ratio {detection.get("ratio", 0):.3f}', 
                                fontsize=12, fontweight='bold')
                axes[0].grid(axis='y', alpha=0.3)
                
                # Add percentage labels
                total_before = sum(counts_before)
                for i, (cls, count) in enumerate(zip(classes, counts_before)):
                    pct = count / total_before * 100
                    axes[0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # After distribution (if SMOTE applied)
            if handling and handling.get("method") == "SMOTE":
                ratio_after = handling.get("ratio_after", {})
                if ratio_after:
                    classes_after = list(ratio_after.keys())
                    # Estimate counts from ratios (approximate)
                    total_after = total_before * 2  # SMOTE doubles minority class
                    counts_after = [int(total_after * ratio_after.get(c, 0)) for c in classes_after]
                    
                    axes[1].bar(classes_after, counts_after, color=['#2ecc71', '#27ae60'], alpha=0.7)
                    axes[1].set_xlabel('Class', fontsize=11, fontweight='bold')
                    axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
                    axes[1].set_title(f'After SMOTE: Balanced', fontsize=12, fontweight='bold')
                    axes[1].grid(axis='y', alpha=0.3)
                    
                    # Add percentage labels
                    for i, (cls, count) in enumerate(zip(classes_after, counts_after)):
                        pct = count / total_after * 100
                        axes[1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
                else:
                    axes[1].text(0.5, 0.5, 'SMOTE Applied\n(Details not available)', 
                               ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            else:
                axes[1].text(0.5, 0.5, 'No balancing applied\nor details not available', 
                           ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            
            plt.tight_layout()
            plot_path = self.output_dir / "class_imbalance_resolution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating imbalance resolution plot: {e}")
            return None
    
    def generate_multicollinearity_removal(self, multicollinearity_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate network graph showing correlated pairs and removals"""
        try:
            detection = multicollinearity_reasoning.get("detection", {})
            removal = multicollinearity_reasoning.get("removal", {})
            
            if not detection or not detection.get("detected"):
                return None
            
            G = nx.Graph()
            removed_features = set()
            
            if removal:
                removed_features = {f["feature"] for f in removal.get("removed_features", [])}
            
            # Add nodes and edges
            high_corr_pairs = detection.get("high_corr_pairs", [])
            for pair in high_corr_pairs[:15]:  # Limit to top 15 pairs
                col1 = pair["col1"]
                col2 = pair["col2"]
                corr = pair["correlation"]
                
                G.add_node(col1)
                G.add_node(col2)
                G.add_edge(col1, col2, weight=corr)
            
            if len(G.nodes()) == 0:
                return None
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Color nodes: red for removed, green for kept
            node_colors = ['#e74c3c' if node in removed_features else '#2ecc71' for node in G.nodes()]
            node_sizes = [800 if node in removed_features else 600 for node in G.nodes()]
            
            # Draw edges with thickness based on correlation
            edges = G.edges(data=True)
            edge_weights = [e[2]['weight'] for e in edges]
            nx.draw_networkx_edges(G, pos, width=[w*3 for w in edge_weights], 
                                 alpha=0.6, edge_color='gray', ax=ax)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                                 alpha=0.8, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
            
            ax.set_title('Multicollinearity Network: Correlated Features\n(Red=Removed, Green=Kept)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
            
            plt.tight_layout()
            plot_path = self.output_dir / "multicollinearity_removal.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating multicollinearity removal plot: {e}")
            return None
    
    def generate_feature_engineering_summary(self, fe_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate summary of feature creation/removal"""
        try:
            removed = fe_reasoning.get("removed", [])
            interactions = fe_reasoning.get("interactions", {})
            temporal = fe_reasoning.get("temporal", {})
            cyclical = fe_reasoning.get("cyclical", {})
            
            if not any([removed, interactions, temporal, cyclical]):
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Removed features
            if removed:
                removed_features = [f["feature"] for f in removed]
                reasons = [f["reason"][:30] for f in removed]  # Truncate
                
                axes[0, 0].barh(removed_features[:10], [1]*min(len(removed_features), 10), 
                              color='#e74c3c', alpha=0.7)
                axes[0, 0].set_title(f'Removed Features ({len(removed_features)})', 
                                    fontsize=11, fontweight='bold')
                axes[0, 0].set_xlabel('Removed', fontsize=10)
            
            # Created interactions
            if interactions:
                interaction_cols = list(interactions.keys())[:10]
                corrs = [interactions[col].get("correlation_with_target", 0) for col in interaction_cols]
                
                axes[0, 1].barh(interaction_cols, corrs, color='#3498db', alpha=0.7)
                axes[0, 1].set_title('Created Interaction Features', fontsize=11, fontweight='bold')
                axes[0, 1].set_xlabel('Correlation with Target', fontsize=10)
            
            # Temporal features
            if temporal:
                temporal_features = temporal.get("created", [])
                axes[1, 0].bar(range(len(temporal_features)), [1]*len(temporal_features), 
                              color='#9b59b6', alpha=0.7)
                axes[1, 0].set_xticks(range(len(temporal_features)))
                axes[1, 0].set_xticklabels(temporal_features, rotation=45, ha='right')
                axes[1, 0].set_title('Temporal Features Created', fontsize=11, fontweight='bold')
                axes[1, 0].set_ylabel('Created', fontsize=10)
            
            # Cyclical encoding
            if cyclical:
                encoded_cols = cyclical.get("columns_encoded", [])
                created_features = cyclical.get("created_features", [])
                axes[1, 1].bar(['Encoded Columns', 'Created Features'], 
                              [len(encoded_cols), len(created_features)], 
                              color='#f39c12', alpha=0.7)
                axes[1, 1].set_title('Cyclical Encoding', fontsize=11, fontweight='bold')
                axes[1, 1].set_ylabel('Count', fontsize=10)
            
            plt.tight_layout()
            plot_path = self.output_dir / "feature_engineering_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating feature engineering summary: {e}")
            return None
    
    def generate_temporal_split(self, temporal_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate timeline showing train/test periods"""
        try:
            detection = temporal_reasoning.get("detection", {})
            split_method = temporal_reasoning.get("split_method", {})
            
            if not detection or not detection.get("detected"):
                return None
            
            date_range = detection.get("date_range", {})
            if not date_range:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Parse dates
            start_date = pd.to_datetime(date_range.get("start", "2007-01-01"))
            end_date = pd.to_datetime(date_range.get("end", "2017-12-31"))
            
            # Calculate split point (80%)
            total_days = (end_date - start_date).days
            split_days = int(total_days * 0.8)
            split_date = start_date + pd.Timedelta(days=split_days)
            
            # Draw timeline
            ax.barh([0], [(split_date - start_date).days], left=0, height=0.5, 
                   color='#3498db', alpha=0.7, label='Train Period')
            ax.barh([0], [(end_date - split_date).days], left=(split_date - start_date).days, 
                   height=0.5, color='#e74c3c', alpha=0.7, label='Test Period')
            
            # Add date labels
            ax.axvline(x=(split_date - start_date).days, color='black', linestyle='--', linewidth=2)
            ax.text((split_date - start_date).days, 0.5, f'Split\n{split_date.strftime("%Y-%m-%d")}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_xlim(0, total_days)
            ax.set_ylim(-0.5, 1)
            ax.set_xlabel('Days from Start', fontsize=11, fontweight='bold')
            ax.set_title(f'Temporal Train-Test Split\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', 
                        fontsize=12, fontweight='bold')
            ax.set_yticks([])
            ax.legend(loc='upper right')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / "temporal_split.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating temporal split plot: {e}")
            return None
    
    def generate_location_patterns(self, location_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate heatmap showing location-specific missing patterns"""
        try:
            if not location_reasoning.get("detected"):
                return None
            
            high_missing = location_reasoning.get("high_missing_locations", {})
            if not high_missing:
                return None
            
            # Sort by missing percentage
            sorted_locations = sorted(high_missing.items(), key=lambda x: x[1], reverse=True)[:15]
            locations = [loc[0] for loc in sorted_locations]
            percentages = [loc[1] for loc in sorted_locations]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap data (single row)
            heatmap_data = np.array([percentages]).T
            
            im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=0, vmax=100)
            
            ax.set_yticks(range(len(locations)))
            ax.set_yticklabels(locations)
            ax.set_xticks([0])
            ax.set_xticklabels(['Missing %'])
            ax.set_title('Location-Specific Missing Data Patterns\n(Top 15 Locations)', 
                        fontsize=12, fontweight='bold', pad=15)
            
            # Add percentage labels
            for i, pct in enumerate(percentages):
                ax.text(0, i, f'{pct:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white' if pct > 50 else 'black')
            
            plt.colorbar(im, ax=ax, label='Missing Percentage')
            plt.tight_layout()
            
            plot_path = self.output_dir / "location_missing_patterns.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating location patterns plot: {e}")
            return None
    
    def generate_outlier_treatment(self, cleaning_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate plot showing outlier treatment (especially Rainfall)"""
        try:
            outlier_cols = []
            outlier_counts = []
            methods = []
            
            for col, col_reasoning in cleaning_reasoning.items():
                if isinstance(col_reasoning, dict) and col_reasoning.get("problem", {}).get("type") == "outliers":
                    outlier_cols.append(col)
                    outlier_counts.append(col_reasoning["problem"].get("count", 0))
                    methods.append(col_reasoning.get("action", {}).get("method", "unknown"))
            
            if not outlier_cols:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(outlier_cols, outlier_counts, color='#e74c3c', alpha=0.7)
            ax.set_xlabel('Outliers Detected', fontsize=11, fontweight='bold')
            ax.set_ylabel('Column', fontsize=11, fontweight='bold')
            ax.set_title('Outlier Detection and Treatment', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add method labels
            for i, (bar, method) in enumerate(zip(bars, methods)):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                       method, ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plot_path = self.output_dir / "outlier_treatment.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating outlier treatment plot: {e}")
            return None
    
    def generate_model_selection_flowchart(self, model_reasoning: Dict[str, Any]) -> Optional[str]:
        """Generate flowchart showing data characteristics → model choice"""
        try:
            selected_model = model_reasoning.get("selected_model") or model_reasoning.get("best_model")
            data_chars = model_reasoning.get("data_characteristics", {})
            reason = model_reasoning.get("reason", "Not specified")
            
            if not selected_model:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            # Create flowchart elements
            y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]
            
            # Data characteristics box
            chars_text = f"Data Characteristics:\n"
            chars_text += f"• Balanced: {data_chars.get('is_balanced', 'N/A')}\n"
            chars_text += f"• Complexity: {data_chars.get('complexity_score', 0):.2f}\n"
            chars_text += f"• Missing: {data_chars.get('missing_percentage', 0):.1f}%"
            
            ax.text(0.5, y_positions[0], chars_text, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3),
                   fontsize=11, fontweight='bold')
            
            # Arrow
            ax.annotate('', xy=(0.5, y_positions[1]+0.05), xytext=(0.5, y_positions[0]-0.05),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            
            # Analysis box
            analysis_text = "Problem Analysis:\n"
            analysis_text += "• Class imbalance detected\n"
            analysis_text += "• High missing values\n"
            analysis_text += "• Multicollinearity present"
            
            ax.text(0.5, y_positions[1], analysis_text, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3),
                   fontsize=11, fontweight='bold')
            
            # Arrow
            ax.annotate('', xy=(0.5, y_positions[2]+0.05), xytext=(0.5, y_positions[1]-0.05),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            
            # Model selection box
            model_text = f"Selected Model:\n{selected_model}\n\n"
            model_text += f"Reason: {reason}"
            
            ax.text(0.5, y_positions[2], model_text, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3),
                   fontsize=12, fontweight='bold')
            
            ax.set_title('Model Selection Reasoning Flowchart', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plot_path = self.output_dir / "model_selection_flowchart.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating model selection flowchart: {e}")
            return None
    
    def generate_problem_solution_matrix(self, reasoning_data: Dict[str, Any]) -> Optional[str]:
        """Generate comprehensive problem-solution matrix"""
        try:
            problems = reasoning_data.get("all_problems", {})
            
            problem_names = []
            solutions = []
            statuses = []
            
            # Map problems to solutions
            problem_solution_map = {
                "class_imbalance": ("SMOTE/Class Weights", reasoning_data.get("imbalance", {}).get("handling", {})),
                "missing_data": ("KNN/Mean/Median Imputation", reasoning_data.get("cleaning", {})),
                "multicollinearity": ("Feature Removal", reasoning_data.get("multicollinearity", {}).get("removal", {})),
                "target_missing": ("Row Removal", reasoning_data.get("cleaning", {}).get("target_nulls", {})),
                "outliers": ("Capping", reasoning_data.get("cleaning", {})),
                "temporal_patterns": ("Time-based Split", reasoning_data.get("temporal", {}).get("split_method", {})),
                "location_missing_patterns": ("Location-aware Imputation", reasoning_data.get("location_patterns", {})),
                "data_leakage": ("Verification", reasoning_data.get("data_leakage", {})),
                "weak_feature_correlations": ("Ensemble Methods", reasoning_data.get("model_selection", {})),
                "high_cardinality": ("Target Encoding", reasoning_data.get("feature_engineering", {}).get("location", {}))
            }
            
            for prob_key, (solution_name, solution_data) in problem_solution_map.items():
                if prob_key in problems and problems[prob_key].get("detected", False):
                    problem_names.append(prob_key.replace("_", " ").title())
                    solutions.append(solution_name)
                    
                    # Check if solution was applied
                    if solution_data:
                        if isinstance(solution_data, dict):
                            if solution_data.get("method") or solution_data.get("removed_features") or solution_data.get("detected"):
                                statuses.append("✓ Resolved")
                            else:
                                statuses.append("⚠ Partial")
                        else:
                            statuses.append("✓ Resolved")
                    else:
                        statuses.append("✗ Not Addressed")
            
            if not problem_names:
                return None
            
            fig, ax = plt.subplots(figsize=(14, max(8, len(problem_names)*0.6)))
            
            # Create matrix
            y_pos = np.arange(len(problem_names))
            
            # Color by status
            colors = []
            for status in statuses:
                if "✓" in status:
                    colors.append('#2ecc71')
                elif "⚠" in status:
                    colors.append('#f39c12')
                else:
                    colors.append('#e74c3c')
            
            bars = ax.barh(y_pos, [1]*len(problem_names), color=colors, alpha=0.7)
            
            # Add solution labels
            for i, (bar, solution, status) in enumerate(zip(bars, solutions, statuses)):
                ax.text(0.5, bar.get_y() + bar.get_height()/2, 
                       f'{solution} - {status}', ha='center', va='center',
                       fontweight='bold', fontsize=10, color='white')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(problem_names)
            ax.set_xlabel('Status', fontsize=11, fontweight='bold')
            ax.set_title('Problem-Solution Matrix: All Critical Issues', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlim(0, 1.2)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / "problem_solution_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error generating problem-solution matrix: {e}")
            return None

