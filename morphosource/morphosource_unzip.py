import os
import zipfile
from pathlib import Path
import pandas as pd
import yaml
# Z:\workspace\goswami-lab\Marco_suture\Morpho_source_specimens\Euarchontoglires\Euarchonta (all Morphosource specimens downloaded)

# Function to unzip a file into a target directory
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
        
    # Initialize a list to store zip file and unzip folder pairs
    

    file_path = './morpho.yaml'
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    input_folder = config['input_folder']
    output_folder = config['output_folder']

    data = []
           
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.zip'):
                is_unzip = True
                error_message = "N/A"
                
                zip_path = Path(root) / file
                print(f'Found ZIP file: {zip_path}')
                try:
                    # Create a directory for unzipped files if it doesn't exist
                    unzip_dir =  os.path.join(output_folder , Path(file).stem)
                    os.makedirs(unzip_dir, exist_ok=True)
                    # Unzip the file to unzip_dir
                    unzip_file(zip_path, unzip_dir)
                except Exception as e:
                    error_message =  str(e)
                    is_unzip = False
                
                data.append({"Zip_File": zip_path, "Unzip_Folder": unzip_dir,
                            "Success": is_unzip, "error_message":error_message})
                ## Go through unzip_dir and find 

    print("Unzipping complete.")


    df = pd.DataFrame(data)

    # save the DataFrame of unzip log.
    unzip_log_path = os.path.join(input_folder , "unzip_log.csv")
    df.to_csv(unzip_log_path,index=False)


    ### Read the unzip log, and extract info, or unzip more files from the unzipped folders
    
    # csv_path = os.path.abspath(os.path.join(start_dir , "unzip_log.csv"))
    # print(f"opening the csv file:{'dataset/'}")
    df = pd.read_csv(unzip_log_path)

    folders = df['Unzip_Folder']
    ori_zips = df['Zip_File']
    unzip_paths = df['Unzip_Folder']
    dfs = []

    file_log = []

    for folder,ori_zip in zip(folders,ori_zips):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.csv'):
                    is_csv = True
                    error_message = "N/A"
                    csv_path = Path(root) / file
                    print(f'Found csv file: {csv_path}')
                    try:
                        df = pd.read_csv(csv_path)
                        
                        new_row = {"Log_Original_zip_file": ori_zip,"Log_Unzip_Folder":folder,"Log_csv_file":csv_path}

                        for col,val in new_row.items():
                            if col not in df.columns:
                                df[col]=val
                        
                        dfs.append(df)

                    except Exception as e:
                        is_csv = False
                        error_message = str(e)
                        # file_log.append({"Original zip file": ori_zip, "Unzip_Folder":folder, "csv_file":csv_path, "Open": False, "Error": str(e)})
                    file_log.append({"File Type": "CSV", "Original zip file": ori_zip,"Unzip_Folder":folder, 
                                    "csv_file":csv_path,
                                    "Open": is_csv,
                                    "Error":error_message})
                if file.endswith('.zip'):
                    is_unzip = True
                    error_message = "N/A"

                    print(f"Found zip file: {file}")
                    zip_path = Path(root) / file
                    try:
                        # Create a directory for unzipped files if it doesn't exist
                        unzip_dir =  os.path.join(folder , Path(file).stem)
                        os.makedirs(unzip_dir, exist_ok=True)
                        # Unzip the file to unzip_dir
                        unzip_file(zip_path, unzip_dir)
                    except Exception as e:
                        error_message =  str(e)
                        is_unzip = False
                    
                    file_log.append({"File Type": "ZIP", 
                                    "Original zip file": ori_zip,"Unzip_Folder":folder, 
                                    "new_zip_file": zip_path, 
                                    "new_unzip_folder":unzip_dir,
                                    "Open": is_unzip,
                                    "Error":error_message})

                        
                    # file_log.append({"csv_file":csv_path})
    all_data = pd.concat(dfs, ignore_index=True)

    meta_path = os.path.abspath(os.path.join(input_folder , "metadata.csv"))
    all_data.to_csv(meta_path,index=False)              


    df = pd.DataFrame(file_log)

    # save the DataFrame of unzip log.
    output_path = os.path.join(input_folder , "read_files_log.csv")
    df.to_csv(output_path,index=False)