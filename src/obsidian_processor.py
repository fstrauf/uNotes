import os
import re
import json
import frontmatter
from pathlib import Path
from typing import Dict, List, Any
import logging

class ObsidianProcessor:
    def __init__(self, vault_path: str, output_dir: str):
        self.vault_path = Path(vault_path)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
    def process_vault(self) -> List[Dict[str, Any]]:
        """Process entire Obsidian vault"""
        self.logger.info(f"Processing vault: {self.vault_path}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all markdown files
        md_files = self._find_markdown_files()
        self.logger.info(f"Found {len(md_files)} markdown files")
        
        # Process files
        processed_notes = []
        for md_file in md_files:
            try:
                note_data = self._process_note(md_file)
                processed_notes.append(note_data)
                
                # Save as text file for GraphRAG
                self._save_for_graphrag(note_data)
                
            except Exception as e:
                self.logger.error(f"Error processing {md_file}: {e}")
        
        return processed_notes
    
    def _find_markdown_files(self) -> List[Path]:
        """Find all markdown files excluding Obsidian system files"""
        md_files = []
        for file_path in self.vault_path.rglob("*.md"):
            if ".obsidian" not in str(file_path):
                md_files.append(file_path)
        return md_files
    
    def _process_note(self, file_path: Path) -> Dict[str, Any]:
        """Process individual markdown note"""
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        # Extract components
        return {
            'filename': file_path.stem,
            'path': str(file_path),
            'folder': file_path.parent.name,
            'frontmatter': post.metadata,
            'content': post.content,
            'wikilinks': self._extract_wikilinks(post.content),
            'tags': self._extract_tags(post.content, post.metadata),
            'headers': self._extract_headers(post.content),
            'text_content': self._clean_content(post.content)
        }
    
    def _extract_wikilinks(self, content: str) -> List[str]:
        """Extract [[wikilinks]] from content"""
        pattern = r'\[\[([^\]]+)\]\]'
        matches = re.findall(pattern, content)
        return [match.split('|')[0].strip() for match in matches]
    
    def _extract_tags(self, content: str, frontmatter: Dict) -> List[str]:
        """Extract tags from content and frontmatter"""
        tags = []
        
        # From frontmatter
        if 'tags' in frontmatter:
            fm_tags = frontmatter['tags']
            if isinstance(fm_tags, str):
                tags.extend([t.strip() for t in fm_tags.split(',')])
            elif isinstance(fm_tags, list):
                tags.extend(fm_tags)
        
        # From content
        inline_tags = re.findall(r'#([a-zA-Z0-9_/-]+)', content)
        tags.extend(inline_tags)
        
        return list(set(tags))
    
    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract markdown headers"""
        headers = []
        for line in content.split('\n'):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                text = line.lstrip('#').strip()
                headers.append({'level': level, 'text': text})
        return headers
    
    def _clean_content(self, content: str) -> str:
        """Clean content for GraphRAG processing"""
        # Remove wikilinks but keep text
        content = re.sub(r'\[\[([^\]|]+)(\|[^\]]+)?\]\]', r'\1', content)
        
        # Convert tags to readable format
        content = re.sub(r'#(\w+)', r'Topic: \1', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _save_for_graphrag(self, note_data: Dict[str, Any]):
        """Save processed note for GraphRAG ingestion"""
        output_file = self.output_dir / 'input' / f"{note_data['filename']}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive text representation
        text_content = f"Title: {note_data['filename']}\n"
        text_content += f"Folder: {note_data['folder']}\n"
        
        if note_data['tags']:
            text_content += f"Tags: {', '.join(note_data['tags'])}\n"
        
        if note_data['frontmatter']:
            # Convert any non-serializable objects to strings
            serializable_metadata = {}
            for key, value in note_data['frontmatter'].items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_metadata[key] = value
                except (TypeError, ValueError):
                    serializable_metadata[key] = str(value)
            text_content += f"Metadata: {json.dumps(serializable_metadata)}\n"
        
        text_content += f"\nContent:\n{note_data['text_content']}"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_content) 