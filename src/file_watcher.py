import time
import logging
from pathlib import Path
from typing import Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

from .obsidian_processor import ObsidianProcessor
from .graphrag_manager import GraphRAGManager

logger = logging.getLogger(__name__)

class ObsidianVaultWatcher(FileSystemEventHandler):
    """Watches an Obsidian vault for changes and triggers incremental updates"""
    
    def __init__(self, vault_path: str, workspace_path: str, update_delay: int = 30):
        self.vault_path = Path(vault_path)
        self.workspace_path = Path(workspace_path)
        self.update_delay = update_delay  # seconds to wait before processing changes
        
        self.processor = ObsidianProcessor(str(vault_path), str(workspace_path))
        self.graphrag_manager = GraphRAGManager(str(workspace_path))
        
        # Track pending changes
        self.pending_changes: Set[Path] = set()
        self.last_change_time: Optional[float] = None
        
        logger.info(f"Watching vault: {vault_path}")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if isinstance(event, FileModifiedEvent) and event.src_path.endswith('.md'):
            self._schedule_update(Path(event.src_path))
    
    def on_created(self, event):
        """Handle file creation events"""
        if isinstance(event, FileCreatedEvent) and event.src_path.endswith('.md'):
            self._schedule_update(Path(event.src_path))
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if isinstance(event, FileDeletedEvent) and event.src_path.endswith('.md'):
            self._schedule_update(Path(event.src_path))
    
    def _schedule_update(self, file_path: Path):
        """Schedule an incremental update"""
        # Skip .obsidian folder
        if '.obsidian' in str(file_path):
            return
        
        self.pending_changes.add(file_path)
        self.last_change_time = time.time()
        
        logger.info(f"Scheduled update for: {file_path.name}")
    
    def _process_pending_changes(self):
        """Process all pending changes"""
        if not self.pending_changes:
            return
        
        logger.info(f"Processing {len(self.pending_changes)} changed files...")
        
        # Process changed files
        for file_path in self.pending_changes:
            if file_path.exists():
                try:
                    note_data = self.processor._process_note(file_path)
                    self.processor._save_for_graphrag(note_data)
                    logger.info(f"Processed: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Run incremental GraphRAG indexing
        try:
            logger.info("Running incremental GraphRAG indexing...")
            success = self.graphrag_manager.run_incremental_indexing()
            if success:
                logger.info("Incremental indexing completed successfully")
            else:
                logger.error("Incremental indexing failed")
        except Exception as e:
            logger.error(f"Error during incremental indexing: {e}")
        
        # Clear pending changes
        self.pending_changes.clear()
        self.last_change_time = None
    
    def check_for_updates(self):
        """Check if it's time to process pending updates"""
        if (self.pending_changes and 
            self.last_change_time and 
            time.time() - self.last_change_time >= self.update_delay):
            
            self._process_pending_changes()

class VaultMonitor:
    """Main class for monitoring vault changes"""
    
    def __init__(self, vault_path: str, workspace_path: str, update_delay: int = 30):
        self.vault_path = vault_path
        self.workspace_path = workspace_path
        self.update_delay = update_delay
        
        self.observer = Observer()
        self.event_handler = ObsidianVaultWatcher(vault_path, workspace_path, update_delay)
        
    def start_monitoring(self):
        """Start monitoring the vault for changes"""
        self.observer.schedule(
            self.event_handler, 
            self.vault_path, 
            recursive=True
        )
        self.observer.start()
        
        logger.info("Started vault monitoring. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
                self.event_handler.check_for_updates()
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring the vault"""
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped vault monitoring") 