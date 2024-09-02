##help codes for avizo console
# addon = hx_project.objects.get("read_addon")
# addon.portnames
import glob
import os
from pathlib import Path
import time
import pandas as pd
import numpy as np
import math
from tifffile import imwrite




# def stack_to_mesh(binary_mask , output_path, downsample_scale=10):

    # verts, faces, normals, values = marching_cubes(binary_mask, level=0.5)
    # # Step 3: Create a Trimesh object
    # mesh = Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # # Simplify the mesh
    # # Set target number of faces (e.g., reduce to 50% of the original number of faces)
    # target_faces = mesh.faces.shape[0] // downsample_scale
    # simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
        
    #     # Step 4: Save the mesh to a file
    
    # simplified_mesh.export(output_path)

def is_island_id(value):
    if np.isnan(value) or (value is None) or (math.isnan(value)):
        return False
    elif value <=0:
        return False
    else:
        return True

def is_valid_path(path_str):
    try:
        path = Path(path_str)
        # Check if the path exists and is either a file or a directory
        if path.exists() and (path.is_file() or path.is_dir()):
            return True
        else:
            return False
    except Exception as e:
        # If any exception occurs, it's not a valid path
        return False
def del_all_objs(objs):
    """Delete objects of from a list of objs

    Args:
        objs (_type_): a list of objs
    """
    for obj in objs:
        hx_project.remove(obj)

def del_all_objs_by_names(names):
    """Delete objects of from a list of objs

    Args:
        objs (_type_): a list of objs
    """
    for name in names:
        try:
            hx_project.remove(hx_project.get(name))
        except Exception as e:
            print("Error in deleting")

def gen_suture_dict(part_id_dict, df_mapping):
    # Convert the row to a dictionary
    # part_id_dict = df.iloc[0].to_dict()

    result = {}
    for index, row in df_mapping.iterrows():
        result[row['result']] = (part_id_dict[row['part_1']],
                                part_id_dict[row['part_2']])
    return result

def auto_find_row_idx(df_id , file_name):
    
    max_prefix_length = 2
    return_row = None
    
    for index, row in df_id.iterrows():
        seg_name = os.path.basename(row['seg_file'])
        prefix_length = common_prefix_length(seg_name, file_name)
        print(seg_name, file_name)
        print(prefix_length)
        if prefix_length > max_prefix_length:
            max_prefix_length = prefix_length
            return_row = row
    
    return return_row

def common_prefix_length(str1, str2):
    # Find the minimum length of the two strings
    min_length = min(len(str1), len(str2))
    
    # Initialize a counter for the common prefix length
    prefix_length = 0
    
    # Iterate through the characters of both strings
    for i in range(min_length):
        if str1[i] == str2[i]:
            prefix_length += 1
        else:
            break
    
    return prefix_length

class GenSutures(PyScriptObject):
    def __init__(self):
        
        self.segs = []
        self.converts = []
        self.files = []
        import _hx_core
        _hx_core._tcl_interp('startLogInhibitor')


        
        self.input = HxConnection(self, "input", " Input for suture generations")
        self.input.valid_types = ('HxUniformLabelField3')


        self.input_template= HxPortFilename(self, "input_template", "CSV file for template")
        self.input_bone_ids= HxPortFilename(self, "input_bone_ids", "CSV file for bone id")
        

        self.recipe_name= HxPortText(self, "recipe_name", "Recipe")
        self.recipe_name.text = "Image Recipe Player"
        
        self.functions = HxPortRadioBox(self,"functions", "Match methods")
        self.functions.radio_boxes = [
            HxPortRadioBox.RadioBox(label="Auto"),
            HxPortRadioBox.RadioBox(label="Specify rows")
            ]
        self.functions.selected = 0
        
        self.row= HxPortText(self, "row", "Row number")
        self.row.text = "0"  
        
        self.output_dir= HxPortFilename(self, "output_dir", "Folder for output")
        self.output_dir.mode = HxPortFilename.SAVE_DIRECTORY
        
        
        
        
        # Classic 'Apply' button:
        self.do_it = HxPortDoIt(self, "apply", "Apply")

        _hx_core._tcl_interp('stopLogInhibitor')

    
    def addon_clean(self):
        try:
            if self.data.visible:
                self.data.visible = False
            if self.ports.startStop.visible:
                self.ports.startStop.visible = False
            if self.ports.showConsole.visible:
                self.ports.showConsole.buttons[0].hit = True
                self.fire()
                self.ports.showConsole.visible = False
        except Exception as e:
            print(f"Waiting for ports being created {e}")
    
    def update(self):
        self.addon_clean()
        
        if self.functions.selected ==0:
            self.row.visible = False
        elif self.functions.selected == 1:
            self.row.visible = True

        pass
    
    
    def compute(self):
        # Does the user press the 'Apply' button?
        if not self.do_it.was_hit:
            return

        print(self.input_bone_ids.filenames)
        print(self.input_template.filenames)
        print(self.recipe_name.text)
        

        try:
            recipe = hx_project.get(self.recipe_name.text)
            df_id = pd.read_csv(self.input_bone_ids.filenames)
            df_template = pd.read_csv(self.input_template.filenames)
        except KeyError:
            hx_message.error("Recipe name doesn't match. Please check")
            return 
        except Exception as e:
            hx_message.error("Please check if input is correct")
            return 


        if self.output_dir.filenames == "" or self.output_dir.filenames is None:
            hx_message.info("No folder specified for output meshes, won't save meshes")

    
    
        ## Get the input
        if self.input.source() is None:
            hx_message.error(f"Please select an input data")
            return
        suture_input = self.input.source()
        input_array = suture_input.get_array()
    
        # Get the corresponding ID
        if self.functions.selected ==1:
            try:
                # Try to convert the string to an integer
                row_idx = int(self.row.text)
                row = df_id.iloc[row_idx,]
            except ValueError:
                # If conversion fails, report an error
                print("Error: The provided string is not an integer.")
                return
        elif self.functions.selected ==0:
            row = auto_find_row_idx(df_id , suture_input.name)
            if row is None:
                hx_message.error(f"Auto row matching failed. Please specify it manually")
                return
        
        print(f"Using row {row}")
        
            
        suture_dict = gen_suture_dict(row, df_template)
        print(suture_dict)
    

        suture_pairs = list(suture_dict.items())
        
        array_all_sutures = np.zeros_like(input_array,dtype='uint8')
        
        # log_dict = {"specimen":row["specimen"]}
        
        recipe_input_names = []
        
        for idx, (key, value) in enumerate(suture_pairs):
            print(f"Processing {key}")

            # output_suture = np.zeros_like(img)
            bone_1_id = value[0]
            bone_2_id = value[1]
            
            if is_island_id(bone_1_id) and is_island_id(bone_2_id) and bone_1_id!=bone_2_id:
        
                array_for_suture = np.zeros_like(input_array,dtype='uint8')
                array_for_suture = np.where(input_array == bone_1_id, 1, array_for_suture)
                array_for_suture = np.where(input_array == bone_2_id, 2, array_for_suture)

                recipe_input = hx_project.create('HxUniformLabelField3')
                recipe_input.name = f"for_{key}_seg"
                recipe_input_names.append(recipe_input.name)
                recipe_input.bounding_box = suture_input.bounding_box
                recipe_input.set_array(array_for_suture)

                recipe.ports.data.connect(recipe_input)

                # set_prev = set(hx_project.objects.keys())
                
                recipe.execute()
                
                # name = list((set(hx_project.objects.keys()).difference(set_prev)))[0]
                # print(f"Generated result:{name}")
                # output = hx_project.get(name)
                
                
                output = recipe.results[0]
                recipe.results.__delitem__(0)
                output_array = output.get_array()

                
                array_all_sutures[output_array == 1] = idx+1
        
        recipe.ports.data.connect(None)
        del_all_objs_by_names(recipe_input_names)

        
        
        recipe_input = hx_project.create('HxUniformLabelField3')
        recipe_input.name = "all_sutures"
        recipe_input.bounding_box = suture_input.bounding_box
        recipe_input.set_array(array_all_sutures)
        
        output_dir = self.output_dir.filenames
        output_path = os.path.join(output_dir, f"sutures_{suture_input.name}.tif")
        
        
        imwrite(output_path,array_all_sutures)