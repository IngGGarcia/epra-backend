"""
Local Storage Client

This module provides a client for handling local file storage operations.
It implements basic file operations like upload, download, and delete.
"""

from pathlib import Path
import shutil
from typing import BinaryIO, Optional


class LocalStorageClient:
    """
    Client for handling local file storage operations.

    This client provides methods to interact with the local filesystem,
    allowing for file uploads, downloads, and deletions.
    """

    def __init__(self, base_path: str = "storage"):
        """
        Initialize the LocalStorageClient.

        Args:
            base_path: The base directory where files will be stored.
                      Defaults to 'storage' in the current working directory.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload(self, file: BinaryIO, destination: str) -> str:
        """
        Upload a file to the local storage.

        Args:
            file: The file object to upload
            destination: The destination path relative to the base_path

        Returns:
            str: The full path where the file was saved

        Raises:
            IOError: If there's an error during file operations
        """
        full_path = self.base_path / destination
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "wb") as f:
            shutil.copyfileobj(file, f)

        return str(full_path)

    def download(self, path: str) -> Optional[BinaryIO]:
        """
        Download a file from the local storage.

        Args:
            path: The path of the file relative to the base_path

        Returns:
            BinaryIO: The file object if found, None otherwise
        """
        full_path = self.base_path / path
        if not full_path.exists():
            return None

        return open(full_path, "rb")

    def delete(self, path: str) -> bool:
        """
        Delete a file from the local storage.

        Args:
            path: The path of the file relative to the base_path

        Returns:
            bool: True if the file was deleted, False otherwise
        """
        full_path = self.base_path / path
        if not full_path.exists():
            return False

        try:
            full_path.unlink()
            return True
        except Exception:
            return False

    def exists(self, path: str) -> bool:
        """
        Check if a file exists in the local storage.

        Args:
            path: The path of the file relative to the base_path

        Returns:
            bool: True if the file exists, False otherwise
        """
        return (self.base_path / path).exists()
