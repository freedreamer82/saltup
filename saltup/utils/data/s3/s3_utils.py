import os
import pathlib
import logging
import boto3
import configparser

from saltup.utils.misc import match_patterns

class S3:
    """
    A class for interacting with an Amazon S3 bucket. This class provides methods to authenticate
    and perform operations such as downloading files and folders from the specified S3 bucket.
    """

    def __init__(self, bucket_name, aws_access_key_id=None, aws_secret_access_key=None, aws_credential_filepath="~/.aws/credentials", section='default'):
        """
        Initializes the S3 client with the specified AWS credentials.

        Args:
            bucket_name (str): The name of the S3 bucket to interact with.
            aws_access_key_id (str, optional): The AWS access key ID. If not provided, 
                                            the credentials file will be used.
            aws_secret_access_key (str, optional): The AWS secret access key. If not provided, 
                                                the credentials file will be used.
            aws_credential_filepath (str, optional): Path to the AWS credentials file to use if 
                                                    access keys are not provided directly. 
                                                    Defaults to "~/.aws/credentials".
            section (str, optional): The section in the credentials file to retrieve credentials 
                                    from. Defaults to 'default'.
        """
        self._aws_access_key_id, self._aws_secret_access_key = self._get_aws_credentials(
            aws_access_key_id, aws_secret_access_key, aws_credential_filepath, section
        )
        self._bucket_name = bucket_name

        # Initialize the S3 client using the retrieved credentials
        self._client = boto3.client(
            's3',
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key
        )

    def _get_aws_credentials(self, aws_access_key_id, aws_secret_access_key, aws_credential_filepath, section='default'):
        """
        Retrieves AWS credentials either from arguments or from the credentials file.

        Args:
            aws_access_key_id (str): The AWS access key ID to use.
            aws_secret_access_key (str): The AWS secret access key to use.
            aws_credential_filepath (str): Path to the AWS credentials file to read from 
                                        if the access keys are not provided as arguments.
            section (str): The section of the credentials file to retrieve credentials from.

        Returns:
            tuple: A tuple containing the AWS access key ID and secret access key.

        Raises:
            ValueError: If credentials are not found in either the arguments or the specified file.
        """
        if aws_access_key_id and aws_secret_access_key:
            return aws_access_key_id, aws_secret_access_key

        if aws_credential_filepath:
            config = configparser.ConfigParser()
            credential_file_path = os.path.expanduser(aws_credential_filepath)
            if os.path.isfile(credential_file_path):
                config.read(credential_file_path)
                return (
                    config.get(section, 'aws_access_key_id'),
                    config.get(section, 'aws_secret_access_key')
                )
            else:
                raise ValueError(f"The specified credential file '{credential_file_path}' does not exist.")
        
        raise ValueError("AWS credentials are required. Provide them either directly as arguments or through a credentials file.")

    def download_file(self, file_path: str, destination_path: str, overwrite: bool = True, retries: int = 3) -> bool:
        """
        Downloads a file from the S3 bucket with retries.

        Args:
            file_path (str): The S3 file path to download.
            destination_path (str): The local path to save the downloaded file.
            overwrite (bool): If True, overwrites the file if it exists. Defaults to True.
            retries (int): Number of retry attempts in case of failure. Defaults to 3.

        Returns:
            bool: True if the download is successful, False otherwise.
        """
        file_name = os.path.basename(file_path)
        local_file_path = os.path.join(destination_path, file_name)

        # Check if the file already exists and overwrite is False
        if os.path.exists(local_file_path) and not overwrite:
            logging.debug(f"File '{local_file_path}' already exists. Skipping download.")
            return True
        
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)

        for attempt in range(retries):
            try:
                # Download the file from S3
                self._client.download_file(self._bucket_name, file_path, local_file_path)
                logging.debug(f"File downloaded at {local_file_path}")
                return True
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed to download file '{file_path}': {e}")
                if attempt == retries - 1:
                    logging.error(f"Failed to download file '{file_path}' after {retries} attempts.")
                    return False

    def download_file_from_folder(self, folder_path: str, destination_path: str, list_filter: list = None, overwrite: bool = True, retries: int = 3):
        """
        Downloads files from a specific folder in the S3 bucket.

        Args:
            folder_path (str): The S3 folder path to download files from.
            destination_path (str): The local path to save downloaded files.
            list_filter (list): A list of file filename patterns to filter which files to download.
            retries (int): Number of retry attempts in case of failure. Defaults to 3.
            overwrite (bool): If True, overwrites existing files. Defaults to True.
        """
        paginator = self._client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self._bucket_name, Prefix=folder_path)

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        for page in pages:
            for obj in page.get('Contents', []):
                file = obj['Key']
                filename = os.path.basename(file)
                
                # Check if the file matches the filter patterns
                if (not list_filter) or match_patterns(filename, list_filter):
                    if not self.download_file(file, destination_path, overwrite=overwrite):
                        logging.warning(f"Failed to download {obj['Key']}.")

    def download_folder(self, s3_folder: str, local_dir: str, list_filter: list = None, overwrite: bool = True, retries: int = 3):
        """
        Downloads a folder from an S3 bucket to a local directory.

        Args:
            s3_folder (str): The folder path in the S3 bucket.
            local_dir (str): The local directory to save the downloaded files.
            list_filter (list): A list of file filename patterns to filter which files to download.
            retries (int): Number of retry attempts in case of failure. Defaults to 3.
            overwrite (bool): If True, overwrites existing files. Defaults to True.
        """
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List objects in the specified S3 folder
        paginator = self._client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self._bucket_name, Prefix=s3_folder)

        # Loop through the objects in the S3 folder
        for page in pages:
            for obj in page.get('Contents', []):
                # Extract the full S3 path of the object
                s3_path = obj['Key']
                if s3_path.endswith("/"):  # Skip if it's a folder marker
                    continue

                # Determine local file path
                relative_path = os.path.relpath(s3_path, s3_folder)
                local_file_path = os.path.join(local_dir, relative_path)

                # Ensure the local directory exists for the file
                local_file_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)

                filename = os.path.basename(s3_path)
                
                # Check if the file matches the filter patterns
                if (not list_filter) or match_patterns(filename, list_filter):
                    if not self.download_file(s3_path, local_file_dir, overwrite=overwrite):
                        logging.warning(f"Failed to download {obj['Key']}.")
    
    def ls(self, s3_folder:str='./', only_basename:bool = True):
        """
        Lists the files in a specified folder in the S3 bucket.

        Args:
            s3_folder (str): The folder path in the S3 bucket to list contents of. Defaults to root ('./').

        Returns:
            list: A list of file paths within the specified S3 folder.
        """
        file_list = []
        paginator = self._client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self._bucket_name, Prefix=s3_folder)

        for page in pages:
            for obj in page.get('Contents', []):
                # Skip folder markers and only add files to the list
                if not obj['Key'].endswith('/'):
                    path = os.path.basename(obj['Key']) if only_basename else obj['Key']
                    file_list.append(path)

        return file_list
