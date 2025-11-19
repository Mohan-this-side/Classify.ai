"""
Flowchart and Diagram Generator
Creates flowcharts and architecture diagrams for reports.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logging.warning("Graphviz not available. Flowcharts will not be generated.")

logger = logging.getLogger(__name__)


class FlowchartGenerator:
    """Generates flowcharts and diagrams."""
    
    def __init__(self, output_dir: str = "Evaluation/results/visualization/diagrams"):
        """Initialize the flowchart generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz not available. Install with: pip install graphviz")
    
    def generate_workflow_state_diagram(
        self,
        output_filename: str = "workflow_state_diagram"
    ):
        """Generate LangGraph workflow state diagram."""
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz not available. Skipping workflow diagram.")
            return
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Classification Workflow')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='rounded')
        
        # Define nodes
        nodes = [
            'Data Cleaning',
            'Data Discovery',
            'EDA Analysis',
            'Feature Engineering',
            'ML Building',
            'Model Evaluation',
            'Technical Reporting',
            'Project Management'
        ]
        
        for node in nodes:
            dot.node(node)
        
        # Define edges (workflow flow)
        edges = [
            ('Data Cleaning', 'Data Discovery'),
            ('Data Discovery', 'EDA Analysis'),
            ('EDA Analysis', 'Feature Engineering'),
            ('Feature Engineering', 'ML Building'),
            ('ML Building', 'Model Evaluation'),
            ('Model Evaluation', 'Technical Reporting'),
            ('Technical Reporting', 'Project Management')
        ]
        
        for edge in edges:
            dot.edge(edge[0], edge[1])
        
        # Save
        output_path = self.output_dir / output_filename
        dot.render(str(output_path), format='png', cleanup=True)
        logger.info(f"Saved workflow state diagram to {output_path}.png")
    
    def generate_architecture_diagram(
        self,
        output_filename: str = "architecture_diagram"
    ):
        """Generate system architecture diagram."""
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz not available. Skipping architecture diagram.")
            return
        
        # Create graph
        dot = graphviz.Digraph(comment='System Architecture')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box')
        
        # Layer 1 (Hardcoded)
        with dot.subgraph(name='cluster_layer1') as layer1:
            layer1.attr(label='Layer 1: Hardcoded Analysis')
            layer1.attr(style='filled')
            layer1.attr(color='lightblue')
            layer1.node('L1_DataCleaning', 'Data Cleaning\n(Hardcoded)')
            layer1.node('L1_EDA', 'EDA\n(Hardcoded)')
            layer1.node('L1_FeatureEng', 'Feature Engineering\n(Hardcoded)')
        
        # Layer 2 (LLM)
        with dot.subgraph(name='cluster_layer2') as layer2:
            layer2.attr(label='Layer 2: LLM-Generated Code')
            layer2.attr(style='filled')
            layer2.attr(color='lightgreen')
            layer2.node('L2_LLM', 'LLM Service')
            layer2.node('L2_Sandbox', 'Sandbox Executor')
            layer2.node('L2_Validator', 'Code Validator')
        
        # Agents
        dot.node('Agents', '8 Agents\n(Data Discovery, EDA, etc.)')
        
        # State Management
        dot.node('State', 'State Management\n(LangGraph)')
        
        # Connections
        dot.edge('Agents', 'L1_DataCleaning')
        dot.edge('Agents', 'L1_EDA')
        dot.edge('Agents', 'L1_FeatureEng')
        dot.edge('Agents', 'L2_LLM')
        dot.edge('L2_LLM', 'L2_Sandbox')
        dot.edge('L2_Sandbox', 'L2_Validator')
        dot.edge('State', 'Agents')
        
        # Save
        output_path = self.output_dir / output_filename
        dot.render(str(output_path), format='png', cleanup=True)
        logger.info(f"Saved architecture diagram to {output_path}.png")
    
    def generate_test_coverage_matrix(
        self,
        test_scenarios: Dict[str, List[str]],
        output_filename: str = "test_coverage_matrix"
    ):
        """Generate test coverage matrix."""
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz not available. Skipping test coverage matrix.")
            return
        
        # Create graph
        dot = graphviz.Digraph(comment='Test Coverage Matrix')
        dot.attr(rankdir='LR')
        
        # Agents
        agents = [
            'Data Discovery', 'EDA', 'Data Cleaning', 'Feature Engineering',
            'ML Builder', 'Model Evaluation', 'Technical Reporter', 'Project Manager'
        ]
        
        # Test scenarios
        scenarios = list(test_scenarios.keys())
        
        # Create nodes
        for agent in agents:
            dot.node(f'agent_{agent}', agent, shape='box')
        
        for scenario in scenarios:
            dot.node(f'scenario_{scenario}', scenario, shape='ellipse', style='filled', color='lightyellow')
        
        # Create edges (which agents are tested by which scenarios)
        for scenario, tested_agents in test_scenarios.items():
            for agent in tested_agents:
                dot.edge(f'scenario_{scenario}', f'agent_{agent}')
        
        # Save
        output_path = self.output_dir / output_filename
        dot.render(str(output_path), format='png', cleanup=True)
        logger.info(f"Saved test coverage matrix to {output_path}.png")
    
    def generate_all_diagrams(self):
        """Generate all diagrams."""
        logger.info("Generating all flowcharts and diagrams...")
        
        self.generate_workflow_state_diagram()
        self.generate_architecture_diagram()
        
        # Test coverage matrix (example)
        test_scenarios = {
            'Class Imbalance Detection': ['EDA', 'ML Builder', 'Model Evaluation'],
            'Anti-Cheating': ['ML Builder', 'Model Evaluation'],
            'Multicollinearity': ['Feature Engineering', 'EDA'],
            'High Dimensionality': ['Data Discovery', 'Feature Engineering']
        }
        self.generate_test_coverage_matrix(test_scenarios)
        
        logger.info("All diagrams generated successfully")

