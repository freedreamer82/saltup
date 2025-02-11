import os
import logging
import boto3
import configparser
import re
from datetime import datetime
from typing import Union, Iterable, List

from saltup.utils.misc import match_patterns

class S3:
    """
    A class for interacting with an Amazon S3 bucket. This class provides methods to authenticate
    and perform operations such as downloading files and folders from the specified S3 bucket.
    """

    def __init__(
        self, 
        bucket_name:str , 
        aws_access_key_id:str =None, 
        aws_secret_access_key:str =None, 
        aws_credential_filepath:str ="~/.aws/credentials", 
        section: str='default'
    ):
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

    def _get_aws_credentials(self, aws_access_key_id:str, aws_secret_access_key:str, aws_credential_filepath:str, section:str):
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
            credential_file_path = os.path.expanduser(aws_credential_filepath)
            if os.path.isfile(credential_file_path):
                return self.read_aws_credentials(credential_file_path, section)
            else:
                raise ValueError(f"Credential file not found: {credential_file_path}")
        
        raise ValueError("AWS credentials are required. Provide them either directly as arguments or through a credentials file.")

    @staticmethod
    def read_aws_credentials(credential_file_path, section='default'):
        """
        Reads AWS credentials from a given file.

        This function attempts to read AWS credentials from a file in two formats:
        1. INI format, where credentials are stored under a specified section.
        2. Plain text format, where credentials are stored as key-value pairs.

        Args:
            credential_file_path (str): The path to the credentials file.
            section (str, optional): The section in the INI file to read the credentials from. Defaults to 'default'.

        Returns:
            tuple: A tuple containing the AWS access key ID and AWS secret access key.

        Raises:
            ValueError: If the credentials format is unrecognized or if the required keys are not found.
        """
        try:
            # Try to read as INI
            config = configparser.ConfigParser()
            config.read(credential_file_path)
            return (
                config.get(section, 'aws_access_key_id'),
                config.get(section, 'aws_secret_access_key')
            )
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback for non-INI formats
            with open(credential_file_path, 'r') as f:
                lines = f.readlines()
            aws_access_key_id = None
            aws_secret_access_key = None
            for line in lines:
                if 'AwsAccessKeyId' in line:
                    aws_access_key_id = line.split('=')[1].strip().strip('"')
                elif 'AwsSecretAccessKey' in line:
                    aws_secret_access_key = line.split('=')[1].strip().strip('"')
            if aws_access_key_id and aws_secret_access_key:
                return aws_access_key_id, aws_secret_access_key
            raise ValueError("Unrecognized credentials format")

    def download_file(
        self, 
        file_path: str, 
        destination_path: str, 
        overwrite: bool = True, 
        retries: int = 3
    ) -> bool:
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

    def download_file_from_folder(
        self, 
        folder_path: str, 
        destination_path: str, 
        patterns: Union[str, Iterable[Union[str, List[str]]]] = None, 
        overwrite: bool = True, 
        retries: int = 3
    ):
        """
        Downloads files from a specific folder in the S3 bucket.

        Args:
            folder_path (str): The S3 folder path to download files from.
            destination_path (str): The local path to save downloaded files.
            patterns: Unix-like patterns to filter the files. Defaults to None. 
                Uses the `saltup.utils.misc.match_patterns()` function to allow for more pattern possibilities.
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
                if (not patterns) or match_patterns(filename, patterns):
                    if not self.download_file(file, destination_path, overwrite=overwrite, retries=retries):
                        logging.warning(f"Failed to download {obj['Key']}.")

    def download_folder(
        self, 
        s3_folder: str, 
        local_dir: str, 
        patterns: Union[str, Iterable[Union[str, List[str]]]] = None, 
        overwrite: bool = True, 
        retries: int = 3
    ):
        """
        Downloads a folder from an S3 bucket to a local directory.

        Args:
            s3_folder (str): The folder path in the S3 bucket.
            local_dir (str): The local directory to save the downloaded files.
            patterns: Unix-like patterns to filter the files. Defaults to None. 
                Uses the `saltup.utils.misc.match_patterns()` function to allow for more pattern possibilities.
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
                if (not patterns) or match_patterns(filename, patterns):
                    if not self.download_file(s3_path, local_file_dir, overwrite=overwrite, retries=retries):
                        logging.warning(f"Failed to download {obj['Key']}.")
    
    def ls(
        self, 
        s3_folder: str = './', 
        patterns: Union[str, Iterable[Union[str, List[str]]]] = None, 
        only_basename: bool = True
    ):
        """
        Lists the files in a specified folder in the S3 bucket.

        Args:
            s3_folder: The folder path in the S3 bucket to list contents of. Defaults to './'.
            patterns: Unix-like patterns to filter the files. Defaults to None. 
                Uses the `saltup.utils.misc.match_patterns()` function to allow for more pattern possibilities.
            only_basename: If True, only the base name of the files will be returned. If False, the full path will be returned. Defaults to True.

        Returns:
            A list of file paths within the specified S3 folder. If `only_basename` is True, the list will contain only the base names of the files. Otherwise, it will contain the full paths.

        Examples:
            # List all files in the root folder
            >>> s3.ls()
            ['file1.txt', 'file2.jpg', 'file3.pdf']

            # List files with a specific extension (e.g., .txt)
            >>> s3.ls(patterns='*.txt')
            ['file1.txt', 'notes.txt']

            # List files matching multiple patterns (OR logic)
            >>> s3.ls(patterns=['*.txt', '*.jpg'])
            ['file1.txt', 'file2.jpg', 'notes.txt']

            # List files matching combined patterns (AND logic)
            >>> s3.ls(patterns=[['data_*', '*.csv']])
            ['data_2023.csv', 'data_2024.csv']

            # List files with full paths instead of just base names
            >>> s3.ls(only_basename=False)
            ['folder/file1.txt', 'folder/file2.jpg']

            # List files in a specific subfolder
            >>> s3.ls(s3_folder='my-folder/')
            ['file1.txt', 'file2.jpg']

            # List files with no patterns (returns all files)
            >>> s3.ls(patterns=None)
            ['file1.txt', 'file2.jpg', 'file3.pdf']

            # List files with an empty patterns list (returns no files)
            >>> s3.ls(patterns=[])
            []
        """
        file_list = []
        paginator = self._client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self._bucket_name, Prefix=s3_folder)

        for page in pages:
            for obj in page.get('Contents', []):
                # Skip folder markers and only add files to the list
                if not obj['Key'].endswith('/'):
                    path = os.path.basename(
                        obj['Key']) if only_basename else obj['Key']

                    if (not patterns) or match_patterns(path, patterns):
                        file_list.append(path)

        return file_list

    def list_files_by_date(
        self,
        s3_folder: str = './',
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        patterns: Union[str, Iterable[Union[str, List[str]]]] = None,
        only_basename: bool = True
    ) -> List[str]:
        """
        List files from an S3 bucket filtered by date range, reusing ls() functionality.
        
        Args:
            s3_folder: The folder path in the S3 bucket to list contents of. Defaults to './'.
            start_date (Union[str, datetime], optional): Start date. If string, format should be '%d.%m.%Y_%H.%M.%SZ'.
            end_date (Union[str, datetime], optional): End date. If string, format should be '%d.%m.%Y_%H.%M.%SZ'.
            patterns: Unix-like patterns to filter the files. Defaults to None.
                Uses the `saltup.utils.misc.match_patterns()` function to allow for more pattern possibilities.
            only_basename: If True, returns only filenames. If False, returns full S3 paths.

        Returns:
            List[str]: List of filtered file paths/names that match the date criteria and patterns.
            
        Example:
            >>> s3.list_files_by_date(
            ...     s3_folder="data/logs/",
            ...     start_date="01.01.2024_00.00.00Z",
            ...     end_date="31.01.2024_23.59.59Z",
            ...     patterns="*.txt"
            ... )
            ['log@15.01.2024_10.30.00Z_data.txt', 'log@16.01.2024_15.45.00Z_data.txt']
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%d.%m.%Y_%H.%M.%SZ')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%d.%m.%Y_%H.%M.%SZ')

        # If no date filtering is needed, just return ls() results
        if not start_date and not end_date:
            return self.ls(s3_folder=s3_folder, patterns=patterns, only_basename=only_basename)

        # Get all files that match the patterns using ls()
        all_files = self.ls(s3_folder=s3_folder, patterns=patterns, only_basename=only_basename)
        
        # Pattern for extracting datetime from filenames
        pattern = r'@(\d{2}\.\d{2}\.\d{4}_\d{2}\.\d{2}\.\d{2}Z)_'
        
        filtered_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            match = re.search(pattern, filename)
            
            if not match:
                continue
                
            file_date = datetime.strptime(match.group(1), '%d.%m.%Y_%H.%M.%SZ')
            
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            
            filtered_files.append(file_path)
        
        return filtered_files
