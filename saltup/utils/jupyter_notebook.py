import datetime
import random
from IPython import get_ipython
import os
import nbformat
import string

# Genera unique ID for the notebook

def generate_notebook_id():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
    random_part = ''.join(random.choices(string.ascii_lowercase, k=4))  # Generates 4 random lower case letters
    return f"{timestamp}_{random_part}"

def save_current_notebook(output_dir='notebook_backup'):
    """
    Saves a copy of the current notebook in the specified directory using nbformat.
    
    Args:
        output_dir (str): Directory where to save the notebook. Default: ‘notebook_backup’
    
    Returns:
        str: Path to the saved file
    """
    
    # Creates the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create file name with timestamp
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"notebook_backup_{timestamp}.ipynb"
    output_path = os.path.join(output_dir, output_file)
    
    # Get the corrent notebook
    ip = get_ipython()
    
    try:
        # Get the contents of the notebook directly
        notebook_content = ip.kernel.shell.user_ns['_ih']
        
        # Create a new notebook
        nb = nbformat.v4.new_notebook()
        
        # Add all non-empty cells
        
        for cell_content in notebook_content:
            if cell_content.strip():
                cell = nbformat.v4.new_code_cell(source=cell_content)
                nb.cells.append(cell)
        
        # Save the notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"Notebook successfully saved in: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error while saving notebook: {str(e)}")
        return None
