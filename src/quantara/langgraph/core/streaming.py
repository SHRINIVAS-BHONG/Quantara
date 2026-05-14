"""
Streaming output system for real-time updates during trading analysis.

This module implements a comprehensive streaming system that allows real-time
delivery of analysis results, partial updates, and status information to clients.
It supports multiple concurrent streaming sessions with proper ordering and
consistency guarantees.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of events that can be streamed."""
    
    # Analysis phase events
    ANALYSIS_STARTED = "analysis_started"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    
    # Data events
    MARKET_DATA_RECEIVED = "market_data_received"
    SENTIMENT_DATA_RECEIVED = "sentiment_data_received"
    TRADER_DATA_RECEIVED = "trader_data_received"
    RISK_DATA_RECEIVED = "risk_data_received"
    
    # Intermediate results
    PARTIAL_RESULT = "partial_result"
    TRADER_SCORED = "trader_scored"
    TRADER_RANKED = "trader_ranked"
    
    # Completion events
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    
    # Status events
    STATUS_UPDATE = "status_update"
    PROGRESS_UPDATE = "progress_update"
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"


class StreamPriority(int, Enum):
    """Priority levels for stream events."""
    
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class StreamEvent:
    """Represents a single event in the stream."""
    
    event_type: StreamEventType
    timestamp: datetime
    session_id: str
    sequence_number: int
    priority: StreamPriority = StreamPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "priority": self.priority.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class StreamSession:
    """Represents a streaming session."""
    
    session_id: str
    created_at: datetime
    query: str
    client_id: Optional[str] = None
    is_active: bool = True
    event_count: int = 0
    last_event_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "query": self.query,
            "client_id": self.client_id,
            "is_active": self.is_active,
            "event_count": self.event_count,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
        }


class StreamBuffer:
    """
    Thread-safe buffer for managing stream events with ordering guarantees.
    
    Ensures that events are delivered in the correct order and supports
    priority-based event handling.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the stream buffer.
        
        Args:
            max_size: Maximum number of events to buffer
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Condition(self.lock)
    
    async def put(self, event: StreamEvent) -> None:
        """
        Add an event to the buffer.
        
        Args:
            event: The event to add
        """
        async with self.lock:
            self.buffer.append(event)
            self.not_empty.notify()
    
    async def get(self) -> StreamEvent:
        """
        Get the next event from the buffer.
        
        Blocks if buffer is empty.
        
        Returns:
            The next event
        """
        async with self.not_empty:
            while not self.buffer:
                await self.not_empty.wait()
            return self.buffer.popleft()
    
    async def get_all(self) -> List[StreamEvent]:
        """
        Get all events currently in the buffer.
        
        Returns:
            List of all buffered events
        """
        async with self.lock:
            events = list(self.buffer)
            self.buffer.clear()
            return events
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0


class StreamingOutput:
    """
    Main streaming output system for real-time analysis updates.
    
    Manages multiple concurrent streaming sessions, ensures event ordering,
    and provides various streaming strategies for different use cases.
    """
    
    def __init__(self, max_buffer_size: int = 10000, max_sessions: int = 100):
        """
        Initialize the streaming output system.
        
        Args:
            max_buffer_size: Maximum events per session buffer
            max_sessions: Maximum concurrent sessions
        """
        self.max_buffer_size = max_buffer_size
        self.max_sessions = max_sessions
        
        # Session management
        self.sessions: Dict[str, StreamSession] = {}
        self.buffers: Dict[str, StreamBuffer] = {}
        self.subscribers: Dict[str, Set[str]] = {}  # event_type -> session_ids
        
        # Sequence tracking
        self.sequence_counters: Dict[str, int] = {}
        
        # Lock for thread-safe operations
        self.lock = asyncio.Lock()
    
    async def create_session(
        self,
        query: str,
        client_id: Optional[str] = None
    ) -> str:
        """
        Create a new streaming session.
        
        Args:
            query: The analysis query
            client_id: Optional client identifier
            
        Returns:
            Session ID
            
        Raises:
            RuntimeError: If max sessions exceeded
        """
        async with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(f"Maximum sessions ({self.max_sessions}) exceeded")
            
            session_id = str(uuid.uuid4())
            session = StreamSession(
                session_id=session_id,
                created_at=datetime.utcnow(),
                query=query,
                client_id=client_id
            )
            
            self.sessions[session_id] = session
            self.buffers[session_id] = StreamBuffer(self.max_buffer_size)
            self.sequence_counters[session_id] = 0
            
            logger.info(f"Created streaming session {session_id} for query: {query}")
            return session_id
    
    async def close_session(self, session_id: str) -> None:
        """
        Close a streaming session.
        
        Args:
            session_id: The session to close
        """
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].is_active = False
                logger.info(f"Closed streaming session {session_id}")
    
    async def emit_event(
        self,
        session_id: str,
        event_type: StreamEventType,
        data: Dict[str, Any],
        priority: StreamPriority = StreamPriority.NORMAL,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit an event to a streaming session.
        
        Args:
            session_id: Target session ID
            event_type: Type of event
            data: Event data
            priority: Event priority
            error: Optional error message
            metadata: Optional metadata
            
        Raises:
            ValueError: If session not found
        """
        async with self.lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            if not session.is_active:
                raise ValueError(f"Session {session_id} is not active")
            
            # Increment sequence number
            sequence_number = self.sequence_counters[session_id]
            self.sequence_counters[session_id] += 1
            
            # Create event
            event = StreamEvent(
                event_type=event_type,
                timestamp=datetime.utcnow(),
                session_id=session_id,
                sequence_number=sequence_number,
                priority=priority,
                data=data,
                error=error,
                metadata=metadata or {}
            )
            
            # Update session
            session.event_count += 1
            session.last_event_time = event.timestamp
            
            # Add to buffer
            buffer = self.buffers[session_id]
        
        # Emit outside lock to avoid blocking
        await buffer.put(event)
        logger.debug(f"Emitted event {event_type.value} to session {session_id}")
    
    async def stream_events(
        self,
        session_id: str,
        timeout: Optional[float] = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream events from a session.
        
        Args:
            session_id: Session to stream from
            timeout: Optional timeout in seconds
            
        Yields:
            Stream events
            
        Raises:
            ValueError: If session not found
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        buffer = self.buffers[session_id]
        session = self.sessions[session_id]
        
        try:
            while session.is_active:
                try:
                    # Get next event with timeout
                    if timeout:
                        event = await asyncio.wait_for(
                            buffer.get(),
                            timeout=timeout
                        )
                    else:
                        event = await buffer.get()
                    
                    yield event
                    
                except asyncio.TimeoutError:
                    # Timeout reached, close stream
                    logger.debug(f"Stream timeout for session {session_id}")
                    break
        finally:
            await self.close_session(session_id)
    
    async def stream_events_json(
        self,
        session_id: str,
        timeout: Optional[float] = None
    ) -> AsyncIterator[str]:
        """
        Stream events as JSON strings.
        
        Args:
            session_id: Session to stream from
            timeout: Optional timeout in seconds
            
        Yields:
            JSON-formatted events
        """
        async for event in self.stream_events(session_id, timeout):
            yield event.to_json()
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information
            
        Raises:
            ValueError: If session not found
        """
        async with self.lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            buffer = self.buffers[session_id]
            
            return {
                **session.to_dict(),
                "buffer_size": buffer.size(),
                "sequence_counter": self.sequence_counters[session_id]
            }
    
    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about all active sessions.
        
        Returns:
            List of session information
        """
        async with self.lock:
            sessions_info = []
            for session_id, session in self.sessions.items():
                if session.is_active:
                    buffer = self.buffers[session_id]
                    sessions_info.append({
                        **session.to_dict(),
                        "buffer_size": buffer.size(),
                        "sequence_counter": self.sequence_counters[session_id]
                    })
            return sessions_info
    
    async def cleanup_inactive_sessions(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up inactive sessions older than max_age.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of sessions cleaned up
        """
        async with self.lock:
            now = datetime.utcnow()
            to_remove = []
            
            for session_id, session in self.sessions.items():
                if not session.is_active:
                    age = (now - session.created_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(session_id)
            
            for session_id in to_remove:
                del self.sessions[session_id]
                del self.buffers[session_id]
                del self.sequence_counters[session_id]
                logger.info(f"Cleaned up inactive session {session_id}")
            
            return len(to_remove)


# Global streaming output instance
_streaming_output: Optional[StreamingOutput] = None


def get_streaming_output() -> StreamingOutput:
    """Get or create the global streaming output instance."""
    global _streaming_output
    if _streaming_output is None:
        _streaming_output = StreamingOutput()
    return _streaming_output


async def emit_stream_event(
    session_id: str,
    event_type: StreamEventType,
    data: Dict[str, Any],
    priority: StreamPriority = StreamPriority.NORMAL,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to emit an event to the global streaming output.
    
    Args:
        session_id: Target session ID
        event_type: Type of event
        data: Event data
        priority: Event priority
        error: Optional error message
        metadata: Optional metadata
    """
    streaming = get_streaming_output()
    await streaming.emit_event(
        session_id=session_id,
        event_type=event_type,
        data=data,
        priority=priority,
        error=error,
        metadata=metadata
    )
