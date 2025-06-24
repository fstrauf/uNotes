import os
import yaml
import subprocess
from pathlib import Path
import logging
from dotenv import load_dotenv

class GraphRAGManager:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.logger = logging.getLogger(__name__)
        load_dotenv()
        
    def initialize_workspace(self):
        """Initialize GraphRAG workspace"""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create settings.yaml
        settings = {
            'encoding_model': 'cl100k_base',
            'skip_workflows': [],
            'llm': {
                'api_key': '${GRAPHRAG_API_KEY}',
                'type': 'openai_chat',
                'model': 'gpt-4o-mini',  # Cost-effective
                'model_supports_json': True,
                'max_tokens': 4000,
                'temperature': 0.0,
                'concurrent_requests': 5,
                'tokens_per_minute': 50000,
                'requests_per_minute': 100
            },
            'embeddings': {
                'api_key': '${GRAPHRAG_API_KEY}',
                'type': 'openai_embedding',
                'model': 'text-embedding-3-small',
                'batch_size': 16
            },
            'chunks': {
                'size': 400,
                'overlap': 50,
                'group_by_columns': ['id']
            },
            'input': {
                'type': 'file',
                'file_type': 'text',
                'base_dir': 'input',
                'file_encoding': 'utf-8',
                'file_pattern': '.*\\.txt$'
            },
            'storage': {
                'type': 'file',
                'base_dir': 'output/${timestamp}/artifacts'
            },
            'entity_extraction': {
                'max_gleanings': 1,
                'summarize_descriptions': True,
                'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION', 
                               'CONCEPT', 'TOPIC', 'PROJECT', 'TOOL', 'DATE']
            },
            'community_reports': {
                'max_length': 1500,
                'max_input_length': 8000
            }
        }
        
        with open(self.workspace_dir / 'settings.yaml', 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)
        
        # Create required directories
        (self.workspace_dir / 'input').mkdir(exist_ok=True)
        (self.workspace_dir / 'output').mkdir(exist_ok=True)
        
        self.logger.info(f"GraphRAG workspace initialized at {self.workspace_dir}")
    
    def run_indexing(self):
        """Execute GraphRAG indexing"""
        self.logger.info("Starting GraphRAG indexing...")
        
        cmd = [
            'python', '-m', 'graphrag',
            'index',
            '--root', str(self.workspace_dir),
            '--verbose'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.workspace_dir)
            
            if result.returncode == 0:
                self.logger.info("GraphRAG indexing completed successfully")
                return True
            else:
                self.logger.error(f"GraphRAG indexing failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running GraphRAG indexing: {e}")
            return False
    
    def run_incremental_indexing(self):
        """Execute incremental GraphRAG indexing for new/changed files"""
        self.logger.info("Starting incremental GraphRAG indexing...")
        
        cmd = [
            'python', '-m', 'graphrag',
            'index',
            '--root', str(self.workspace_dir),
            '--verbose',
            '--update'  # Enable incremental update mode
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.workspace_dir)
            
            if result.returncode == 0:
                self.logger.info("Incremental GraphRAG indexing completed successfully")
                return True
            else:
                self.logger.error(f"Incremental GraphRAG indexing failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running incremental GraphRAG indexing: {e}")
            return False
    
    def query_global(self, question: str) -> str:
        """Execute global GraphRAG query"""
        cmd = [
            '/Users/fstrauf/01_code/uNotes/myenv/bin/python', '-m', 'graphrag',
            'query',
            '--root', str(self.workspace_dir),
            '--method', 'global',
            '--query', question
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=self.workspace_dir)
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error executing query: {e}"
    
    def query_local(self, question: str) -> str:
        """Execute local GraphRAG query"""
        cmd = [
            '/Users/fstrauf/01_code/uNotes/myenv/bin/python', '-m', 'graphrag',
            'query',
            '--root', str(self.workspace_dir),
            '--method', 'local',
            '--query', question
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=self.workspace_dir)
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error executing query: {e}" 