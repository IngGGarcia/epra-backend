"""
This module provides a generic storage manager for handling file uploads
and organizing them into structured directories.

The StorageManager class abstracts the file storage mechanism, allowing for
easy adaptation to cloud storage solutions like AWS S3 or Google Cloud Storage in the future.
"""

from pathlib import Path
import shutil
import time

from fastapi import UploadFile


class StorageManager:
    """
    A generic storage manager for handling file uploads and organizing them into structured directories.

    This class abstracts the file storage mechanism, allowing for easy adaptation to cloud storage
    solutions like AWS S3 or Google Cloud Storage in the future.
    """

    ALLOWED_EXTENSIONS = {".csv"}  # Allowed file types

    def __init__(
        self, base_storage_path: str = "uploads", sub_storage_path: str | None = None
    ):
        """
        Initializes the StorageManager with a specified storage directory.

        Args:
            base_storage_path (str): The root directory where files will be stored.
            sub_storage_path (str, optional): Subdirectory inside the base storage path.
        """
        self.base_storage_path = Path(base_storage_path)
        if sub_storage_path:
            self.base_storage_path = self.base_storage_path / sub_storage_path
        self.base_storage_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the storage directory exists

    def validate_file(self, file_obj: UploadFile) -> bool:
        """
        Validates the uploaded file based on its extension.

        Args:
            file_obj (UploadFile): File to be validated.

        Returns:
            bool: True if the file is valid.

        Raises:
            ValueError: If the file format is not allowed.
        """
        file_extension = Path(file_obj.filename).suffix.lower()
        if file_extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file type: {file_obj.filename}. Allowed: {self.ALLOWED_EXTENSIONS}"
            )

        return True

    def store_files(
        self,
        user_id: int,
        files: dict[str, UploadFile],
        date: str | None = None,
    ) -> dict[str, str]:
        """
        Stores multiple files in a structured directory format.

        Args:
            user_id (int): Unique identifier of the user.
            date (str, optional): Date of the session (format: YYYY-MM-DD). Defaults to current date.
            files (dict[str, UploadFile]): Dictionary where keys are file names and values are UploadFile objects.

        Returns:
            dict[str, str]: A dictionary containing the paths of stored files.

        Raises:
            ValueError: If a file is invalid.
            FileNotFoundError: If there is an issue accessing the storage directory.
            PermissionError: If the system does not have permission to write files.
            RuntimeError: If any other unexpected error occurs.
        """
        try:
            # Default to current date if none is provided
            if date is None:
                date = time.strftime("%Y-%m-%d")

            # Define the structured storage directory
            user_folder: Path = self.base_storage_path / f"user_{user_id}" / date
            user_folder.mkdir(parents=True, exist_ok=True)

            stored_paths = {}

            for filename, file_obj in files.items():
                # Validate the file
                self.validate_file(file_obj)

                file_path: Path = user_folder / filename

                # Store the file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file_obj.file, buffer)

                stored_paths[filename] = str(file_path)

            return stored_paths

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Storage directory not found: {self.base_storage_path}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied while writing to: {self.base_storage_path}"
            ) from e
        except ValueError as e:
            raise ValueError(f"File validation error: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Error storing files: {str(e)}") from e
