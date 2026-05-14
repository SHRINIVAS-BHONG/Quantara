"""
Checkpointing and recovery system for the LangGraph trading system.

This module implements comprehensive checkpointing capabilities including:
- Persistent storage of workflow state
- Automatic checkpoint creation at workflow phases
- Checkpoint validation and integrity checks
- Recovery mechanisms from corrupted checkpoints
- Multiple storage backend support (memory, file, redis, postgres)

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import asyncio
import json
import hashlib
import logging
import gzip
import pickle
from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import os

from .state import TradingState, WorkflowStep


class CheckpointStorageType(Enum):
    """Supported checkpoint storage backends."""
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    POSTGRES = "postgres"


class CheckpointStatus(Enum):
    """Checkpoint status enumeration."""
    VALID = "valid"
    CORRUPTED = "corrupted"
    EXPIRED = "expired"
    INCOMPLETE = "incomplete"


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    created_at: datetime
    workflow_step: str
    state_version: int
    data_hash: str
    compressed: bool
    size_bytes: int
    storage_backend: str
    expiry_time: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "workflow_step": self.workflow_step,
            "state_version": self.state_version,
            "data_hash": self.data_hash,
            "compressed": self.compressed,
            "size_bytes": self.size_bytes,
            "storage_backend": self.storage_backend,
            "expiry_time": self.expiry_time.isoformat() if self.expiry_time else None,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create metadata from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            workflow_step=data["workflow_step"],
            state_version=data["state_version"],
            data_hash=data["data_hash"],
            compressed=data["compressed"],
            size_bytes=data["size_bytes"],
            storage_backend=data["storage_backend"],
            expiry_time=datetime.fromisoformat(data["expiry_time"]) if data.get("expiry_time") else None,
            tags=data.get("tags", [])
        )


@dataclass
class Checkpoint:
    """Complete checkpoint with state and metadata."""
    metadata: CheckpointMetadata
    state: TradingState
    
    def validate_integrity(self) -> bool:
        """
        Validate checkpoint integrity by comparing stored hash with computed hash.
        
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Compute current state hash
            state_data = {k: v for k, v in self.state.items() if not k.startswith("_")}
            state_json = json.dumps(state_data, sort_keys=True)
            computed_hash = hashlib.sha256(state_json.encode()).hexdigest()
            
            # Compare with stored hash
            return computed_hash == self.metadata.data_hash
            
        except Exception:
            return False
    
    def get_status(self) -> CheckpointStatus:
        """
        Get checkpoint status.
        
        Returns:
            CheckpointStatus indicating checkpoint validity
        """
        # Check if expired
        if self.metadata.expiry_time and datetime.now() > self.metadata.expiry_time:
            return CheckpointStatus.EXPIRED
        
        # Check integrity
        if not self.validate_integrity():
            return CheckpointStatus.CORRUPTED
        
        # Check completeness
        required_fields = ["original_query", "current_step", "completed_steps"]
        if not all(field in self.state for field in required_fields):
            return CheckpointStatus.INCOMPLETE
        
        return CheckpointStatus.VALID


class CheckpointStorage:
    """Base class for checkpoint storage backends."""
    
    async def save(self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata) -> bool:
        """Save checkpoint data."""
        raise NotImplementedError
    
    async def load(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint data."""
        raise NotImplementedError
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint."""
        raise NotImplementedError
    
    async def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        raise NotImplementedError
    
    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints."""
        raise NotImplementedError


class MemoryCheckpointStorage(CheckpointStorage):
    """In-memory checkpoint storage (for testing and development)."""
    
    def __init__(self):
        self._storage: Dict[str, bytes] = {}
        self._metadata: Dict[str, CheckpointMetadata] = {}
        self._logger = logging.getLogger(__name__)
    
    async def save(self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata) -> bool:
        """Save checkpoint to memory."""
        try:
            self._storage[checkpoint_id] = data
            self._metadata[checkpoint_id] = metadata
            self._logger.debug(f"Saved checkpoint {checkpoint_id} to memory")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save checkpoint to memory: {e}")
            return False
    
    async def load(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint from memory."""
        return self._storage.get(checkpoint_id)
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        try:
            if checkpoint_id in self._storage:
                del self._storage[checkpoint_id]
                del self._metadata[checkpoint_id]
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to delete checkpoint from memory: {e}")
            return False
    
    async def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all checkpoints in memory."""
        return list(self._metadata.values())
    
    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints from memory."""
        now = datetime.now()
        expired_ids = [
            cp_id for cp_id, metadata in self._metadata.items()
            if metadata.expiry_time and now > metadata.expiry_time
        ]
        
        for cp_id in expired_ids:
            await self.delete(cp_id)
        
        return len(expired_ids)


class FileCheckpointStorage(CheckpointStorage):
    """File-based checkpoint storage."""
    
    def __init__(self, storage_dir: str = ".checkpoints"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for checkpoint."""
        return self.storage_dir / f"{checkpoint_id}.checkpoint"
    
    def _get_metadata_path(self, checkpoint_id: str) -> Path:
        """Get file path for checkpoint metadata."""
        return self.storage_dir / f"{checkpoint_id}.metadata.json"
    
    async def save(self, checkpoint_id: str, data: bytes, metadata: CheckpointMetadata) -> bool:
        """Save checkpoint to file."""
        try:
            # Ensure storage directory exists
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint data
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            with open(checkpoint_path, 'wb') as f:
                f.write(data)
            
            # Save metadata
            metadata_path = self._get_metadata_path(checkpoint_id)
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            self._logger.debug(f"Saved checkpoint {checkpoint_id} to file")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save checkpoint to file: {e}")
            return False
    
    async def load(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint from file."""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            if not checkpoint_path.exists():
                return None
            
            with open(checkpoint_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            self._logger.error(f"Failed to load checkpoint from file: {e}")
            return None
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from file system."""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            metadata_path = self._get_metadata_path(checkpoint_id)
            
            deleted = False
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                deleted = True
            
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            self._logger.error(f"Failed to delete checkpoint from file: {e}")
            return False
    
    async def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all checkpoints in storage directory."""
        checkpoints = []
        
        try:
            for metadata_file in self.storage_dir.glob("*.metadata.json"):
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = CheckpointMetadata.from_dict(metadata_dict)
                    checkpoints.append(metadata)
        
        except Exception as e:
            self._logger.error(f"Failed to list checkpoints: {e}")
        
        return checkpoints
    
    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints from file system."""
        now = datetime.now()
        expired_count = 0
        
        try:
            checkpoints = await self.list_checkpoints()
            
            for metadata in checkpoints:
                if metadata.expiry_time and now > metadata.expiry_time:
                    if await self.delete(metadata.checkpoint_id):
                        expired_count += 1
        
        except Exception as e:
            self._logger.error(f"Failed to cleanup expired checkpoints: {e}")
        
        return expired_count


class CheckpointManager:
    """
    Comprehensive checkpoint manager for the LangGraph trading system.
    
    Provides automatic checkpointing, validation, recovery, and storage management
    with support for multiple storage backends.
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    
    def __init__(
        self,
        storage_type: CheckpointStorageType = CheckpointStorageType.FILE,
        storage_config: Optional[Dict[str, Any]] = None,
        retention_hours: int = 24,
        compression_enabled: bool = True,
        auto_cleanup: bool = True
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            storage_type: Type of storage backend to use
            storage_config: Configuration for storage backend
            retention_hours: Hours to retain checkpoints before expiry
            compression_enabled: Whether to compress checkpoint data
            auto_cleanup: Whether to automatically cleanup expired checkpoints
        """
        self.storage_type = storage_type
        self.storage_config = storage_config or {}
        self.retention_hours = retention_hours
        self.compression_enabled = compression_enabled
        self.auto_cleanup = auto_cleanup
        
        self._logger = logging.getLogger(__name__)
        self._storage = self._initialize_storage()
        self._checkpoint_history: List[str] = []
        self._max_history_size = 100
        
        # Statistics
        self._stats = {
            "checkpoints_created": 0,
            "checkpoints_restored": 0,
            "checkpoints_failed": 0,
            "corrupted_checkpoints": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
    
    def _initialize_storage(self) -> CheckpointStorage:
        """Initialize the appropriate storage backend."""
        if self.storage_type == CheckpointStorageType.MEMORY:
            return MemoryCheckpointStorage()
        
        elif self.storage_type == CheckpointStorageType.FILE:
            storage_dir = self.storage_config.get("storage_dir", ".checkpoints")
            return FileCheckpointStorage(storage_dir)
        
        elif self.storage_type == CheckpointStorageType.REDIS:
            # TODO: Implement Redis storage backend
            self._logger.warning("Redis storage not yet implemented, falling back to file storage")
            return FileCheckpointStorage()
        
        elif self.storage_type == CheckpointStorageType.POSTGRES:
            # TODO: Implement PostgreSQL storage backend
            self._logger.warning("PostgreSQL storage not yet implemented, falling back to file storage")
            return FileCheckpointStorage()
        
        else:
            self._logger.warning(f"Unknown storage type {self.storage_type}, using file storage")
            return FileCheckpointStorage()
    
    def _generate_checkpoint_id(self, state: TradingState) -> str:
        """
        Generate unique checkpoint ID based on state content.
        
        Args:
            state: TradingState to generate ID for
            
        Returns:
            Unique checkpoint ID
        """
        timestamp = datetime.now().isoformat()
        query = state.get("original_query", "")
        step = state.get("current_step", "")
        
        id_content = f"{timestamp}:{query}:{step}"
        id_hash = hashlib.md5(id_content.encode()).hexdigest()[:16]
        
        return f"checkpoint_{id_hash}_{int(datetime.now().timestamp())}"
    
    def _compute_state_hash(self, state: TradingState) -> str:
        """
        Compute hash of state for integrity checking.
        
        Args:
            state: TradingState to hash
            
        Returns:
            SHA-256 hash of state
        """
        # Extract only non-private fields for hashing
        state_data = {k: v for k, v in state.items() if not k.startswith("_")}
        state_json = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def _serialize_state(self, state: TradingState) -> bytes:
        """
        Serialize state to bytes with optional compression.
        
        Args:
            state: TradingState to serialize
            
        Returns:
            Serialized state as bytes
        """
        # Use pickle for serialization to handle complex types
        state_bytes = pickle.dumps(state)
        
        if self.compression_enabled:
            # Compress with gzip
            state_bytes = gzip.compress(state_bytes)
        
        return state_bytes
    
    def _deserialize_state(self, data: bytes, compressed: bool) -> TradingState:
        """
        Deserialize state from bytes.
        
        Args:
            data: Serialized state bytes
            compressed: Whether data is compressed
            
        Returns:
            Deserialized TradingState
        """
        if compressed:
            data = gzip.decompress(data)
        
        return pickle.loads(data)
    
    async def create_checkpoint(
        self,
        state: TradingState,
        workflow_step: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a checkpoint of the current state.
        
        Args:
            state: TradingState to checkpoint
            workflow_step: Optional workflow step identifier
            tags: Optional tags for checkpoint categorization
            
        Returns:
            Checkpoint ID if successful, None otherwise
            
        Requirements: 4.1, 4.2
        """
        try:
            # Generate checkpoint ID
            checkpoint_id = self._generate_checkpoint_id(state)
            
            # Determine workflow step
            if workflow_step is None:
                workflow_step = state.get("current_step", "unknown")
            
            # Serialize state
            state_bytes = self._serialize_state(state)
            
            # Compute state hash for integrity
            state_hash = self._compute_state_hash(state)
            
            # Calculate expiry time
            expiry_time = datetime.now() + timedelta(hours=self.retention_hours)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                created_at=datetime.now(),
                workflow_step=workflow_step,
                state_version=state.get("_debug_info").state_version if state.get("_debug_info") else 0,
                data_hash=state_hash,
                compressed=self.compression_enabled,
                size_bytes=len(state_bytes),
                storage_backend=self.storage_type.value,
                expiry_time=expiry_time,
                tags=tags or []
            )
            
            # Save to storage
            success = await self._storage.save(checkpoint_id, state_bytes, metadata)
            
            if success:
                # Update state with checkpoint ID
                state["state_checkpoint_id"] = checkpoint_id
                
                # Track in history
                self._checkpoint_history.append(checkpoint_id)
                if len(self._checkpoint_history) > self._max_history_size:
                    self._checkpoint_history.pop(0)
                
                # Update statistics
                self._stats["checkpoints_created"] += 1
                
                self._logger.info(f"Created checkpoint {checkpoint_id} at step {workflow_step}")
                
                # Auto cleanup if enabled
                if self.auto_cleanup:
                    asyncio.create_task(self._storage.cleanup_expired())
                
                return checkpoint_id
            else:
                self._stats["checkpoints_failed"] += 1
                self._logger.error(f"Failed to save checkpoint {checkpoint_id}")
                return None
                
        except Exception as e:
            self._stats["checkpoints_failed"] += 1
            self._logger.error(f"Error creating checkpoint: {e}")
            return None
    
    async def restore_checkpoint(
        self,
        checkpoint_id: str,
        validate: bool = True
    ) -> Optional[TradingState]:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            validate: Whether to validate checkpoint integrity
            
        Returns:
            Restored TradingState if successful, None otherwise
            
        Requirements: 4.3, 4.4
        """
        try:
            self._stats["recovery_attempts"] += 1
            
            # Load checkpoint data
            checkpoint_data = await self._storage.load(checkpoint_id)
            
            if checkpoint_data is None:
                self._logger.error(f"Checkpoint {checkpoint_id} not found")
                return None
            
            # Load metadata
            checkpoints = await self._storage.list_checkpoints()
            metadata = next((cp for cp in checkpoints if cp.checkpoint_id == checkpoint_id), None)
            
            if metadata is None:
                self._logger.error(f"Metadata for checkpoint {checkpoint_id} not found")
                return None
            
            # Deserialize state
            state = self._deserialize_state(checkpoint_data, metadata.compressed)
            
            # Create checkpoint object for validation
            checkpoint = Checkpoint(metadata=metadata, state=state)
            
            # Validate if requested
            if validate:
                status = checkpoint.get_status()
                
                if status == CheckpointStatus.CORRUPTED:
                    self._stats["corrupted_checkpoints"] += 1
                    self._logger.error(f"Checkpoint {checkpoint_id} is corrupted")
                    
                    # Attempt recovery from previous checkpoint
                    return await self._attempt_recovery(checkpoint_id)
                
                elif status == CheckpointStatus.EXPIRED:
                    self._logger.warning(f"Checkpoint {checkpoint_id} is expired")
                    # Still allow restoration of expired checkpoints
                
                elif status == CheckpointStatus.INCOMPLETE:
                    self._logger.warning(f"Checkpoint {checkpoint_id} is incomplete")
                    # Still allow restoration but log warning
            
            # Update statistics
            self._stats["checkpoints_restored"] += 1
            self._stats["successful_recoveries"] += 1
            
            self._logger.info(f"Restored checkpoint {checkpoint_id}")
            
            return state
            
        except Exception as e:
            self._logger.error(f"Error restoring checkpoint {checkpoint_id}: {e}")
            return None
    
    async def _attempt_recovery(self, failed_checkpoint_id: str) -> Optional[TradingState]:
        """
        Attempt to recover from a corrupted checkpoint by finding previous valid checkpoint.
        
        Args:
            failed_checkpoint_id: ID of the corrupted checkpoint
            
        Returns:
            Recovered TradingState if successful, None otherwise
            
        Requirements: 4.4, 4.5
        """
        try:
            self._logger.info(f"Attempting recovery from corrupted checkpoint {failed_checkpoint_id}")
            
            # Find index of failed checkpoint in history
            if failed_checkpoint_id not in self._checkpoint_history:
                self._logger.warning("Failed checkpoint not in history, trying latest checkpoint")
                return await self.get_latest_checkpoint()
            
            failed_index = self._checkpoint_history.index(failed_checkpoint_id)
            
            # Try previous checkpoints in reverse order
            for i in range(failed_index - 1, -1, -1):
                previous_checkpoint_id = self._checkpoint_history[i]
                
                self._logger.info(f"Trying previous checkpoint {previous_checkpoint_id}")
                
                # Try to restore previous checkpoint
                state = await self.restore_checkpoint(previous_checkpoint_id, validate=True)
                
                if state is not None:
                    self._logger.info(f"Successfully recovered from checkpoint {previous_checkpoint_id}")
                    return state
            
            # If no previous checkpoint worked, return None
            self._logger.error("Could not recover from any previous checkpoint")
            return None
            
        except Exception as e:
            self._logger.error(f"Error during recovery attempt: {e}")
            return None
    
    async def validate_checkpoint(self, checkpoint_id: str) -> CheckpointStatus:
        """
        Validate a checkpoint without restoring it.
        
        Args:
            checkpoint_id: ID of checkpoint to validate
            
        Returns:
            CheckpointStatus indicating validation result
            
        Requirements: 4.3
        """
        try:
            # Load checkpoint
            checkpoint_data = await self._storage.load(checkpoint_id)
            
            if checkpoint_data is None:
                return CheckpointStatus.CORRUPTED
            
            # Load metadata
            checkpoints = await self._storage.list_checkpoints()
            metadata = next((cp for cp in checkpoints if cp.checkpoint_id == checkpoint_id), None)
            
            if metadata is None:
                return CheckpointStatus.CORRUPTED
            
            # Deserialize state
            state = self._deserialize_state(checkpoint_data, metadata.compressed)
            
            # Create checkpoint and get status
            checkpoint = Checkpoint(metadata=metadata, state=state)
            return checkpoint.get_status()
            
        except Exception as e:
            self._logger.error(f"Error validating checkpoint {checkpoint_id}: {e}")
            return CheckpointStatus.CORRUPTED
    
    async def get_latest_checkpoint(self) -> Optional[TradingState]:
        """
        Get the most recent valid checkpoint.
        
        Returns:
            Latest valid TradingState if available, None otherwise
        """
        try:
            if not self._checkpoint_history:
                self._logger.warning("No checkpoints in history")
                return None
            
            # Try checkpoints from most recent to oldest
            for checkpoint_id in reversed(self._checkpoint_history):
                state = await self.restore_checkpoint(checkpoint_id, validate=True)
                if state is not None:
                    return state
            
            return None
            
        except Exception as e:
            self._logger.error(f"Error getting latest checkpoint: {e}")
            return None
    
    async def list_checkpoints(
        self,
        workflow_step: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[CheckpointMetadata]:
        """
        List available checkpoints with optional filtering.
        
        Args:
            workflow_step: Optional filter by workflow step
            tags: Optional filter by tags
            
        Returns:
            List of checkpoint metadata
        """
        try:
            checkpoints = await self._storage.list_checkpoints()
            
            # Apply filters
            if workflow_step:
                checkpoints = [cp for cp in checkpoints if cp.workflow_step == workflow_step]
            
            if tags:
                checkpoints = [
                    cp for cp in checkpoints
                    if any(tag in cp.tags for tag in tags)
                ]
            
            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
            
            return checkpoints
            
        except Exception as e:
            self._logger.error(f"Error listing checkpoints: {e}")
            return []
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            success = await self._storage.delete(checkpoint_id)
            
            if success and checkpoint_id in self._checkpoint_history:
                self._checkpoint_history.remove(checkpoint_id)
            
            return success
            
        except Exception as e:
            self._logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False
    
    async def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        try:
            checkpoints = await self._storage.list_checkpoints()
            
            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
            
            # Delete old checkpoints
            deleted_count = 0
            for checkpoint in checkpoints[keep_count:]:
                if await self.delete_checkpoint(checkpoint.checkpoint_id):
                    deleted_count += 1
            
            self._logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
            
        except Exception as e:
            self._logger.error(f"Error cleaning up old checkpoints: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpoint manager statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            "storage_type": self.storage_type.value,
            "compression_enabled": self.compression_enabled,
            "retention_hours": self.retention_hours,
            "checkpoint_history_size": len(self._checkpoint_history),
            "auto_cleanup": self.auto_cleanup
        }


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(
    storage_type: CheckpointStorageType = CheckpointStorageType.FILE,
    storage_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> CheckpointManager:
    """
    Get or create the global checkpoint manager instance.
    
    Args:
        storage_type: Type of storage backend
        storage_config: Configuration for storage backend
        **kwargs: Additional configuration options
        
    Returns:
        CheckpointManager instance
    """
    global _checkpoint_manager
    
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(
            storage_type=storage_type,
            storage_config=storage_config,
            **kwargs
        )
    
    return _checkpoint_manager
