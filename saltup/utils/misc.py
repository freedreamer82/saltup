from typing import Union, List, Iterable, Optional, Tuple
import fnmatch
import shutil
import os
from tqdm import tqdm
from saltup.utils import configure_logging
from contextlib import contextmanager
import sys


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
            

def match_patterns(target: str, patterns: Iterable[Union[str, List[str]]]) -> bool:
    """
    Match a string against single or combined wildcard patterns.

    Args:
        target: String to check against patterns
        patterns: Patterns to match. Can be either:
            - String patterns (matched with OR logic)
            - Lists of patterns (matched with AND logic)

    Returns:
        bool: True if string matches pattern conditions

    Examples:
        >>> match_patterns('test.txt', ['*.txt', 'test*'])  # OR matching
        True
        >>> match_patterns('test.py', [['test*', '*.py']])  # AND matching
        True
    """
    def _check_pattern(pattern: Union[str, List[str]]) -> bool:
        if isinstance(pattern, str):
            return fnmatch.fnmatch(target, pattern)

        if isinstance(pattern, list):
            return all(fnmatch.fnmatch(target, sub_pattern) for sub_pattern in pattern)

        raise TypeError(f"Invalid pattern type: {type(pattern)}. Expected str or list[str].")

    try:
        return any(_check_pattern(pattern) for pattern in patterns)
    except TypeError as e:
        raise TypeError("Invalid pattern in match_patterns") from e


def count_files(root_dir: str, filters: Optional[List[str]] = None, recursive: bool = True) -> Tuple[int, List[str]]:
    """
    Count and list files in a directory matching given filters.

    Args:
        root_dir (str): Root directory to search in.
        filters (Optional[List[str]], optional): List of file patterns to match.
            Supports wildcards like '*' and '?'. Defaults to None (match all files).
        recursive (bool, optional): Whether to search subdirectories. Defaults to True.

    Returns:
        Tuple[int, List[str]]: A tuple containing the count of matched files
        and a list of their relative paths.

    Raises:
        FileNotFoundError: If the root directory does not exist.
        NotADirectoryError: If the root path is not a directory.
    """
    # Validate input directory
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Path is not a directory: {root_dir}")

    # Normalize filters and provide a default that matches everything
    filters = filters or ['*']

    # Collect files based on recursive flag
    if recursive:
        founded = [
            os.path.relpath(os.path.join(dirpath, filename), root_dir)
            for dirpath, _, filenames in os.walk(root_dir)
            for filename in filenames
            if match_patterns(filename, filters)
        ]
    else:
        founded = [
            filename
            for filename in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, filename)) and
               match_patterns(filename, filters)
        ]

    return len(founded), founded


def remove_directories_containing(base_path: str, search_string: str) -> None:
    """
    Recursively remove directories whose names contain a specific string.
    Linux equivalent: find /path -type d -name "*string*" -exec rm -rf {} +

    Args:
        base_path: Starting directory for search
        search_string: String to match in directory names
    """
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if search_string in dir_name:
                full_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(full_path)
                    print(f"Deleted directory: {full_path}")
                except Exception as e:
                    print(f"Error deleting {full_path}: {e}")


def delete_empty_directories(base_path: str) -> None:
    """
    Remove all empty directories under the specified path.
    Linux equivalent: find /path -type d -empty -delete

    Args:
        base_path: Root directory to start deletion from
    """
    # Walk through the directory tree from bottom to top
    for root, dirs, files in os.walk(base_path, topdown=False):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            try:
                # Check if directory is empty
                if not os.listdir(full_path):
                    os.rmdir(full_path)
                    print(f"Deleted empty directory: {full_path}")
            except Exception as e:
                print(f"Error deleting {full_path}: {e}")


def rename_files_with_prefix(folder_path: str, prefix: str, extensions: tuple[str] = (".jpg", ".png", ".txt")) -> None:
    """Add prefix to filenames with specified extensions.

    Args:
        folder_path: Directory containing files
        prefix: String to prepend to filenames
        extensions: File extensions to process
    
    Raises:
        FileNotFoundError: If folder_path doesn't exist
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    modified_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(extensions):
            name, ext = os.path.splitext(filename)
            if name.startswith(prefix):
                continue
            
            new_name = f"{prefix}{name}{ext}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            os.rename(old_path, new_path)
            modified_count += 1
               
    return modified_count


def remove_substring_from_filenames(folder_path: str, substring: str, extensions: tuple[str] = (".jpg", ".png", ".txt")) -> int:
   """Remove substring from filenames with matching extensions.
   
   Args:
       folder_path: Directory to process recursively
       substring: Text to remove from filenames 
       extensions: File extensions to process
   
   Returns:
       Number of files modified
       
   Raises:
       FileNotFoundError: If folder_path not found
   """
   if not os.path.exists(folder_path):
       raise FileNotFoundError(f"Directory not found: {folder_path}")

   modified_count = 0
   for root, _, filenames in os.walk(folder_path):
       for filename in filenames:
           if substring in filename and filename.endswith(extensions):
               old_path = os.path.join(root, filename)
               new_path = os.path.join(root, filename.replace(substring, ""))
               os.rename(old_path, new_path)
               modified_count += 1
               
   return modified_count


def copy_or_move_files(source_folder: str, destination_folder: str, move: bool = False, 
                      extensions: Optional[tuple[str]] = None) -> None:
    """Copy or move files between directories.

    Args:
        source_folder: Source directory containing files
        destination_folder: Target directory
        move: If True, move files instead of copying
        extensions: File extensions to process (None for all files)
    
    Raises:
        FileNotFoundError: If source_folder doesn't exist
    """
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source directory not found: {source_folder}")

    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path) and (extensions is None or filename.endswith(extensions)):
            destination_path = os.path.join(destination_folder, filename)
            if move:
                shutil.move(file_path, destination_path)
            else:
                shutil.copy2(file_path, destination_path)


def unify_files(
    source_dirs: List[str],
    destination: str,
    filters: Optional[List[str]] = None,
    divide_by_extension: bool = False,
    move_files: bool = False,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    Consolidate files from multiple directories into a single destination.

    Args:
        source_dirs: Source directories containing files
        destination: Target directory for consolidated files
        filters: File patterns to include (e.g., ['*.txt', '*.pdf'])
        divide_by_extension: Create subdirectories by file extension
        move_files: Move files instead of copying
        verbose: Enable detailed progress logging

    Returns:
        tuple: (processed_files_count, failed_operations_count)

    Raises:
        ValueError: If no valid source directories found
    """
    logger = configure_logging.get_logger(__name__)
    operation = "Moving" if move_files else "Copying"

    # Validate and collect source directories
    valid_sources = [src for src in source_dirs if os.path.exists(src)]
    if not valid_sources:
        raise ValueError("No valid source directories provided")

    # Collect matching files
    filepaths = []
    for source in valid_sources:
        for root, _, filenames in os.walk(source):
            filepaths.extend(
                os.path.join(root, filename)
                for filename in filenames
                if not filters or match_patterns(filename, filters)
            )

    if not filepaths:
        logger.warning("No files found matching the criteria")
        return 0, 0

    # Prepare destination directory
    os.makedirs(destination, exist_ok=True)
    processed_count = failed_count = 0

    if verbose:
       configure_logging.enable_tqdm()

    # Process files with optional progress bar
    files_iter = tqdm(filepaths, desc=f"{operation} files", disable=not verbose)
    for filepath in files_iter:
        try:
            # Determine destination path
            if divide_by_extension:
                ext = os.path.splitext(filepath)[1][1:] or 'no_extension'
                dest_dir = os.path.join(destination, ext)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                dest_dir = destination

            dest_path = os.path.join(dest_dir, os.path.basename(filepath))

            # Handle filename conflicts
            counter = 1
            while os.path.exists(dest_path):
                base, ext = os.path.splitext(dest_path)
                dest_path = f"{base}_{counter}{ext}"
                counter += 1

            # Process file operation
            if move_files:
                shutil.move(filepath, dest_path)
            else:
                shutil.copy2(filepath, dest_path)

            processed_count += 1

            if verbose:
                logger.debug(f"{operation} file {filepath} to {dest_path}")

        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to {operation.lower()} {filepath}: {e}")

    if verbose:
        configure_logging.disable_tqdm()

    logger.info(f"Operation completed: {processed_count} processed, {failed_count} failed")
    return processed_count, failed_count