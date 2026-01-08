"""Timeout utilities for MCP tools - prevents freezing at all layers."""

import asyncio
import functools
import logging
import time
from typing import Callable, Any, TypeVar, Coroutine

logger = logging.getLogger(__name__)

# Global timeout for any MCP tool (seconds)
MCP_TOOL_TIMEOUT = 60

# Shorter timeout for quick operations
MCP_QUICK_TIMEOUT = 30

T = TypeVar('T')


class MCPTimeoutError(Exception):
    """Raised when an MCP tool times out."""
    pass


def with_timeout(timeout: int = MCP_TOOL_TIMEOUT):
    """Decorator to add timeout protection to async MCP tools.
    
    This provides an outer timeout layer that will kill any tool
    that exceeds the specified duration, preventing MCP freezes.
    
    Usage:
        @mcp.tool()
        @with_timeout(30)
        async def my_tool(...):
            ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start = time.time()
            func_name = func.__name__
            
            logger.info(f"[MCP] Starting {func_name} (timeout={timeout}s)")
            
            try:
                # Run with asyncio timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
                elapsed = time.time() - start
                logger.info(f"[MCP] {func_name} completed in {elapsed:.1f}s")
                return result
                
            except asyncio.TimeoutError:
                elapsed = time.time() - start
                logger.error(f"[MCP] {func_name} TIMEOUT after {elapsed:.1f}s")
                return f"❌ TIMEOUT: {func_name} did not complete within {timeout}s\n\nThis tool was automatically stopped to prevent freezing.\nTry running the operation in background with nohup."
            
            except asyncio.CancelledError:
                elapsed = time.time() - start
                logger.warning(f"[MCP] {func_name} CANCELLED after {elapsed:.1f}s")
                return f"❌ CANCELLED: {func_name} was stopped after {elapsed:.1f}s"
            
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[MCP] {func_name} ERROR after {elapsed:.1f}s: {e}")
                return f"❌ ERROR in {func_name}: {str(e)[:200]}\n\nElapsed: {elapsed:.1f}s"
        
        return wrapper
    return decorator


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: int = MCP_TOOL_TIMEOUT,
    operation_name: str = "operation"
) -> T:
    """Run a coroutine with timeout protection.
    
    Use this for individual async operations within a tool.
    
    Args:
        coro: The coroutine to run
        timeout: Timeout in seconds
        operation_name: Name for logging
        
    Returns:
        The coroutine result, or an error message string
    """
    start = time.time()
    
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        logger.error(f"[MCP] {operation_name} TIMEOUT after {elapsed:.1f}s")
        raise MCPTimeoutError(f"{operation_name} timed out after {timeout}s")
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[MCP] {operation_name} ERROR after {elapsed:.1f}s: {e}")
        raise


def safe_return(func: Callable[..., Coroutine[Any, Any, str]]) -> Callable[..., Coroutine[Any, Any, str]]:
    """Decorator that ensures MCP tools always return a string, never crash.
    
    This catches ALL exceptions and returns them as error messages,
    preventing the MCP server from hanging on uncaught exceptions.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> str:
        start = time.time()
        func_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            
            # Ensure we always return a string
            if result is None:
                return f"✅ {func_name} completed (no output)"
            if not isinstance(result, str):
                return str(result)
            return result
            
        except Exception as e:
            elapsed = time.time() - start
            logger.exception(f"[MCP] {func_name} CRASHED after {elapsed:.1f}s")
            return f"❌ {func_name} CRASHED: {type(e).__name__}: {str(e)[:300]}\n\nElapsed: {elapsed:.1f}s\nPlease report this error."
    
    return wrapper


# Combined decorator for maximum protection
def mcp_safe(timeout: int = MCP_TOOL_TIMEOUT):
    """Combined decorator: timeout + safe return + logging.
    
    Apply this to ALL MCP tools for maximum freeze protection.
    
    Usage:
        @mcp.tool()
        @mcp_safe(timeout=30)
        async def my_tool(...):
            ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, str]]) -> Callable[..., Coroutine[Any, Any, str]]:
        # Apply both decorators
        return with_timeout(timeout)(safe_return(func))
    return decorator
