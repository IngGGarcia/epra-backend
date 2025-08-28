"""
Storage Clients

This module provides clients for different storage backends.
Currently supports:
- Local Storage
- S3 Storage (to be implemented)
"""

from .local import LocalStorageClient
from .local_image import LocalImageClient

__all__ = ["LocalStorageClient", "LocalImageClient"]
