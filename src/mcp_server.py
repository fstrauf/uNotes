import os
import json
import logging
from typing import Dict, Any, List
from fastmcp import FastMCP
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graphrag_manager import GraphRAGManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Universal Knowledge Context Server")

# Global GraphRAG manager instance
graphrag_manager = None

def initialize_graphrag(workspace_path: str):
    """Initialize GraphRAG manager"""
    global graphrag_manager
    graphrag_manager = GraphRAGManager(workspace_path)
    logger.info(f"GraphRAG manager initialized with workspace: {workspace_path}")

@mcp.tool()
def search_knowledge_global(query: str, max_length: int = 2000) -> str:
    """
    Perform global search across the entire knowledge base using community-based reasoning.
    Best for broad questions, thematic analysis, and understanding overall patterns.
    
    Args:
        query: The question or topic to search for
        max_length: Maximum length of response
    
    Returns:
        Comprehensive answer based on community analysis
    """
    if not graphrag_manager:
        return "Error: GraphRAG manager not initialized"
    
    logger.info(f"Global search query: {query}")
    
    try:
        result = graphrag_manager.query_global(query)
        
        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length] + "..."
        
        return result
        
    except Exception as e:
        logger.error(f"Error in global search: {e}")
        return f"Error executing global search: {e}"

@mcp.tool()
def search_knowledge_local(query: str, max_length: int = 2000) -> str:
    """
    Perform local search for specific entities and their immediate relationships.
    Best for precise questions about specific topics, people, or concepts.
    
    Args:
        query: The specific question or entity to search for
        max_length: Maximum length of response
    
    Returns:
        Detailed answer focused on specific entities and relationships
    """
    if not graphrag_manager:
        return "Error: GraphRAG manager not initialized"
    
    logger.info(f"Local search query: {query}")
    
    try:
        result = graphrag_manager.query_local(query)
        
        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length] + "..."
        
        return result
        
    except Exception as e:
        logger.error(f"Error in local search: {e}")
        return f"Error executing local search: {e}"

@mcp.tool()
def get_knowledge_stats() -> str:
    """
    Get statistics about the knowledge base including number of entities,
    relationships, and communities detected.
    
    Returns:
        JSON string with knowledge base statistics
    """
    if not graphrag_manager:
        return "Error: GraphRAG manager not initialized"
    
    try:
        # Read artifacts to get statistics
        workspace_path = Path(graphrag_manager.workspace_dir)
        artifacts_path = workspace_path / "output"
        
        # Find latest artifacts directory
        latest_dir = None
        for item in artifacts_path.iterdir():
            if item.is_dir():
                latest_dir = item / "artifacts"
                break
        
        if not latest_dir or not latest_dir.exists():
            return "No artifacts found. Please run indexing first."
        
        stats = {}
        
        # Read entities
        entities_file = latest_dir / "create_final_entities.parquet"
        if entities_file.exists():
            import pandas as pd
            entities_df = pd.read_parquet(entities_file)
            stats['entities_count'] = len(entities_df)
            stats['entity_types'] = entities_df['type'].value_counts().to_dict()
        
        # Read relationships
        relationships_file = latest_dir / "create_final_relationships.parquet"
        if relationships_file.exists():
            import pandas as pd
            relationships_df = pd.read_parquet(relationships_file)
            stats['relationships_count'] = len(relationships_df)
        
        # Read communities
        communities_file = latest_dir / "create_final_communities.parquet"
        if communities_file.exists():
            import pandas as pd
            communities_df = pd.read_parquet(communities_file)
            stats['communities_count'] = len(communities_df)
        
        return json.dumps(stats, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        return f"Error getting statistics: {e}"

@mcp.resource("knowledge://workspace/status")
def get_workspace_status() -> str:
    """Get current workspace status and configuration"""
    if not graphrag_manager:
        return "GraphRAG manager not initialized"
    
    workspace_path = Path(graphrag_manager.workspace_dir)
    
    status = {
        'workspace_path': str(workspace_path),
        'input_files': len(list((workspace_path / 'input').glob('*.txt'))) if (workspace_path / 'input').exists() else 0,
        'indexed': (workspace_path / 'output').exists() and any((workspace_path / 'output').iterdir()),
        'settings_file': (workspace_path / 'settings.yaml').exists()
    }
    
    return json.dumps(status, indent=2)

def run_mcp_server(workspace_path: str):
    """Run the MCP server"""
    # Initialize GraphRAG
    initialize_graphrag(workspace_path)
    
    # Start server
    logger.info("Starting Universal Knowledge Context MCP Server...")
    mcp.run()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python mcp_server.py <workspace_path>")
        sys.exit(1)
    
    workspace_path = sys.argv[1]
    run_mcp_server(workspace_path) 