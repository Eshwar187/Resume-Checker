"""
Supabase client configuration and utilities.
"""

from supabase import create_client, Client
from supabase.client import ClientOptions
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Supabase client wrapper with connection management."""
    
    def __init__(self):
        self._client: Client = None
        self._service_client: Client = None
    
    @property
    def client(self) -> Client:
        """Get the regular Supabase client (with anon key)."""
        if self._client is None:
            try:
                self._client = create_client(
                    settings.supabase_url,
                    settings.supabase_anon_key,
                    options=ClientOptions(
                        postgrest_client_timeout=10,
                        storage_client_timeout=10
                    )
                )
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                raise
        return self._client
    
    @property
    def service_client(self) -> Client:
        """Get the service role Supabase client (admin privileges)."""
        if self._service_client is None:
            try:
                self._service_client = create_client(
                    settings.supabase_url,
                    settings.supabase_service_role_key,
                    options=ClientOptions(
                        postgrest_client_timeout=10,
                        storage_client_timeout=10
                    )
                )
                logger.info("Supabase service client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase service client: {e}")
                raise
        return self._service_client


# Global Supabase client instance
supabase_client = SupabaseClient()


def get_supabase_client() -> Client:
    """Get the Supabase client instance."""
    return supabase_client.client


def get_supabase_service_client() -> Client:
    """Get the Supabase service client instance."""
    return supabase_client.service_client


class StorageManager:
    """Manage file uploads and downloads with Supabase Storage."""
    
    def __init__(self, bucket_name: str = "resume-files"):
        self.bucket_name = bucket_name
        self.client = get_supabase_service_client()
    
    async def upload_file(
        self, 
        file_path: str, 
        file_content: bytes, 
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to Supabase Storage.
        
        Args:
            file_path: The path where the file will be stored
            file_content: The file content as bytes
            content_type: The MIME type of the file
        
        Returns:
            The public URL of the uploaded file
        """
        try:
            # Upload file
            result = self.client.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=file_content,
                file_options={"content-type": content_type}
            )
            
            if result.error:
                raise Exception(f"Upload failed: {result.error}")
            
            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(file_path)
            
            logger.info(f"File uploaded successfully: {file_path}")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            raise
    
    async def download_file(self, file_path: str) -> bytes:
        """
        Download a file from Supabase Storage.
        
        Args:
            file_path: The path of the file to download
        
        Returns:
            The file content as bytes
        """
        try:
            result = self.client.storage.from_(self.bucket_name).download(file_path)
            
            if result.error:
                raise Exception(f"Download failed: {result.error}")
            
            logger.info(f"File downloaded successfully: {file_path}")
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to download file {file_path}: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from Supabase Storage.
        
        Args:
            file_path: The path of the file to delete
        
        Returns:
            True if deletion was successful
        """
        try:
            result = self.client.storage.from_(self.bucket_name).remove([file_path])
            
            if result.error:
                raise Exception(f"Delete failed: {result.error}")
            
            logger.info(f"File deleted successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            raise
    
    def get_file_url(self, file_path: str) -> str:
        """Get the public URL for a file."""
        return self.client.storage.from_(self.bucket_name).get_public_url(file_path)


# Global storage manager instance
storage_manager = StorageManager()


def get_storage_manager() -> StorageManager:
    """Get the storage manager instance."""
    return storage_manager