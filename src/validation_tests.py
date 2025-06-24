import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

class ValidationTester:
    def __init__(self, processed_notes: List[Dict], graphrag_manager):
        self.processed_notes = processed_notes
        self.graphrag_manager = graphrag_manager
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.setup_baseline()
    
    def setup_baseline(self):
        """Setup baseline semantic search"""
        texts = [note.get('text_content', '') for note in self.processed_notes]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.note_filenames = [note['filename'] for note in self.processed_notes]
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Baseline semantic search using TF-IDF"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.note_filenames[i], similarities[i]) for i in top_indices]
        
        return results
    
    def run_comparative_test(self, test_queries: List[str]) -> Dict:
        """Run comparative test between GraphRAG and semantic search"""
        results = {
            'graphrag_global': [],
            'graphrag_local': [],
            'semantic_search': [],
            'response_times': {'graphrag_global': [], 'graphrag_local': [], 'semantic': []}
        }
        
        for query in test_queries:
            # Test GraphRAG Global
            start_time = time.time()
            global_result = self.graphrag_manager.query_global(query)
            global_time = time.time() - start_time
            
            # Test GraphRAG Local
            start_time = time.time()
            local_result = self.graphrag_manager.query_local(query)
            local_time = time.time() - start_time
            
            # Test Semantic Search
            start_time = time.time()
            semantic_result = self.semantic_search(query)
            semantic_time = time.time() - start_time
            
            # Store results
            results['graphrag_global'].append({
                'query': query,
                'response': global_result,
                'response_time': global_time
            })
            
            results['graphrag_local'].append({
                'query': query,
                'response': local_result,
                'response_time': local_time
            })
            
            results['semantic_search'].append({
                'query': query,
                'response': semantic_result,
                'response_time': semantic_time
            })
        
        return results

# Test scenarios for validation
TEST_QUERIES = [
    # Simple fact retrieval
    "What projects are mentioned in my notes?",
    
    # Multi-hop reasoning
    "How are my different projects connected to each other?",
    
    # Thematic analysis
    "What are the main themes and topics in my knowledge base?",
    
    # Temporal reasoning
    "What activities or events are mentioned with dates?",
    
    # Entity relationships
    "What tools or technologies do I use most frequently?",
    
    # Complex synthesis
    "What insights can you derive from my learning and project patterns?"
] 