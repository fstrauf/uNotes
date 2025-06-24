#!/usr/bin/env python3
"""
Universal Personal Knowledge Context System
Main entry point for processing Obsidian vaults and running knowledge graph analysis
"""

import os
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.obsidian_processor import ObsidianProcessor
from src.graphrag_manager import GraphRAGManager
from src.validation_tests import ValidationTester, TEST_QUERIES
from src.file_watcher import VaultMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_env_vars():
    """Check required environment variables"""
    load_dotenv()
    
    required_vars = ['GRAPHRAG_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please create a .env file based on .env.example")
        return False
    
    return True

def process_vault(vault_path: str, workspace_path: str):
    """Process Obsidian vault and prepare for GraphRAG"""
    logger.info("Starting vault processing...")
    
    processor = ObsidianProcessor(vault_path, workspace_path)
    processed_notes = processor.process_vault()
    
    logger.info(f"Processed {len(processed_notes)} notes")
    return processed_notes

def setup_graphrag(workspace_path: str):
    """Setup and initialize GraphRAG workspace"""
    logger.info("Setting up GraphRAG workspace...")
    
    manager = GraphRAGManager(workspace_path)
    manager.initialize_workspace()
    
    return manager

def run_indexing(manager: GraphRAGManager):
    """Run GraphRAG indexing"""
    logger.info("Running GraphRAG indexing...")
    
    success = manager.run_indexing()
    if success:
        logger.info("Indexing completed successfully")
    else:
        logger.error("Indexing failed")
    
    return success

def run_validation(processed_notes, manager):
    """Run validation tests"""
    logger.info("Running validation tests...")
    
    tester = ValidationTester(processed_notes, manager)
    results = tester.run_comparative_test(TEST_QUERIES)
    
    # Print summary
    print("\n=== VALIDATION RESULTS ===")
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"\nQuery {i+1}: {query}")
        print(f"GraphRAG Global time: {results['graphrag_global'][i]['response_time']:.2f}s")
        print(f"GraphRAG Local time: {results['graphrag_local'][i]['response_time']:.2f}s")
        print(f"Semantic search time: {results['semantic_search'][i]['response_time']:.2f}s")
    
    return results

def interactive_query(manager: GraphRAGManager):
    """Interactive query interface"""
    print("\n=== INTERACTIVE QUERY MODE ===")
    print("Enter 'quit' to exit")
    print("Use 'global:' prefix for global queries or 'local:' for local queries")
    print("Example: global: What are the main themes in my notes?")
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if query.startswith('global:'):
            query = query[7:].strip()
            result = manager.query_global(query)
            print(f"\nGlobal Search Result:\n{result}")
        elif query.startswith('local:'):
            query = query[6:].strip()
            result = manager.query_local(query)
            print(f"\nLocal Search Result:\n{result}")
        else:
            # Default to global search
            result = manager.query_global(query)
            print(f"\nGlobal Search Result:\n{result}")

def main():
    parser = argparse.ArgumentParser(description='Universal Personal Knowledge Context System')
    parser.add_argument('--vault-path', help='Path to Obsidian vault (defaults to OBSIDIAN_VAULT_PATH env var)')
    parser.add_argument('--workspace', default='./data', help='GraphRAG workspace path')
    parser.add_argument('--skip-processing', action='store_true', help='Skip vault processing')
    parser.add_argument('--skip-indexing', action='store_true', help='Skip GraphRAG indexing')
    parser.add_argument('--run-validation', action='store_true', help='Run validation tests')
    parser.add_argument('--interactive', action='store_true', help='Start interactive query mode')
    parser.add_argument('--monitor', action='store_true', help='Start continuous vault monitoring')
    parser.add_argument('--update-delay', type=int, default=30, help='Seconds to wait before processing changes (default: 30)')
    
    args = parser.parse_args()
    
    # Check environment
    if not check_env_vars():
        return 1
    
    # Get vault path from args or environment
    vault_path_str = args.vault_path or os.getenv('OBSIDIAN_VAULT_PATH')
    if not vault_path_str:
        logger.error("No vault path provided. Use --vault-path or set OBSIDIAN_VAULT_PATH in .env")
        return 1
    
    # Validate paths
    vault_path = Path(vault_path_str)
    if not vault_path.exists():
        logger.error(f"Vault path does not exist: {vault_path}")
        logger.info(f"Trying to access: {vault_path_str}")
        return 1
    
    workspace_path = Path(args.workspace)
    processed_notes = []
    
    try:
        # Process vault
        if not args.skip_processing:
            processed_notes = process_vault(str(vault_path), str(workspace_path))
        
        # Setup GraphRAG
        manager = setup_graphrag(str(workspace_path))
        
        # Run indexing
        if not args.skip_indexing:
            if not run_indexing(manager):
                logger.error("Indexing failed, exiting")
                return 1
        
        # Run validation
        if args.run_validation and processed_notes:
            run_validation(processed_notes, manager)
        
        # Interactive mode
        if args.interactive:
            interactive_query(manager)
        
        # Monitoring mode
        if args.monitor:
            monitor = VaultMonitor(str(vault_path), str(workspace_path), args.update_delay)
            monitor.start_monitoring()
        
        logger.info("Process completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 