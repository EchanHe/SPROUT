##help codes for avizo console
# addon = hx_project.objects.get("read_addon")
# addon.portnames
import glob
import os,re
from pathlib import Path
import time
import traceback
import numpy as np

def merge_masks_with_filter(mask1, mask2, ids_to_keep1, ids_to_keep2):
    # Ensure both masks are numpy arrays
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)

    # Create a mask to filter out unwanted IDs
    mask1_filtered = np.where(np.isin(mask1, ids_to_keep1), mask1, 0)
    mask2_filtered = np.where(np.isin(mask2, ids_to_keep2), mask2, 0)

    # Find the maximum ID in the filtered masks
    max_id1 = np.max(mask1_filtered)
    offset = max_id1

    # Offset the IDs in the second mask
    mask2_filtered_offset = np.where(mask2_filtered != 0, mask2_filtered + offset, 0)

    # Combine the masks
    merged_mask = np.where(mask1_filtered != 0, mask1_filtered, mask2_filtered_offset)

    return merged_mask

def replace_values_in_array(arr, values_to_replace, target_value):
    try:
        # Check if the input is a NumPy array
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input is not a NumPy array")

        # Check if the array is 3D
        if arr.ndim != 3:
            raise ValueError("Input array is not 3-dimensional")

        # Check if values_to_replace is a list
        if not isinstance(values_to_replace, list):
            raise ValueError("values_to_replace should be a list")

        # Perform the replacement
        for value in values_to_replace:
            arr[arr == value] = target_value

        return arr
    except Exception as e:
        print(f"Error processing the array: {e}")
        traceback.print_exc()
        return None

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

def is_valid_src_ids_list(s):
    # Define the regex pattern
    pattern = r'^\s*\d+\s*(,\s*\d+\s*)*$'
    # Match the pattern against the input string
    match = re.match(pattern, s)
    return bool(match)        

def is_valid_dst_ids_list(s):
    # Define the regex pattern
    pattern = r'^\s*\d+\s*$'
    # Match the pattern against the input string
    match = re.match(pattern, s)
    return bool(match)     

def string_to_integer_list(s):
    # if is_valid_integer_list(s):
    try:
        # Split the string by commas and strip spaces
        return [int(num.strip()) for num in s.split(',')]
    except:
        raise ValueError("Invalid input string")

class Edit(PyScriptObject):
    def __init__(self):
        
        self.segs = []
        self.converts = []
        self.files = []
        import _hx_core
        _hx_core._tcl_interp('startLogInhibitor')

        # Input port: it will only handle HxUniformScalarField3:
        # self.input = HxConnection(self, "input", "Input Volume")
        # self.input.valid_types = ('HxUniformScalarField3')

        # Input port: it will only handle HxUniformScalarField3:



        # This port will number of segments size.
        # self.n = HxPortIntSlider(self, "n", "Number of Segments")
        # self.n.clamp_range = (1, 100)
        # self.n.value = 14
        
        self.functions = HxPortRadioBox(self,"functions", "How to edit?")
        self.functions.radio_boxes = [
            HxPortRadioBox.RadioBox(label="Merge ids on one image"),
        HxPortRadioBox.RadioBox(label="Merge ids from two image")]
        self.functions.selected = 0
        
        
        self.input_img = HxConnection(self, "input_img", "Input input_img")
        self.input_seed1.valid_types = ('HxUniformLabelField3')
        
        self.delete_ids= HxPortText(self, "delete_ids", "IDs for making original images as background")
        
        self.input_seed = HxConnection(self, "input_seed", "Input seed")
        self.input_seed.valid_types = ('HxUniformLabelField3')
        
        self.src_ids= HxPortText(self, "src_ids", "IDs to change")
        
        self.dst_ids= HxPortText(self, "dst_ids", "Target ID")
        
        
        self.input_seed1 = HxConnection(self, "input_seed1", "Input seed 1")
        self.input_seed1.valid_types = ('HxUniformLabelField3')
        
        self.input_seed2 = HxConnection(self, "input_seed2", "Input seed 2")
        self.input_seed2.valid_types = ('HxUniformLabelField3')
        
        self.keep_id_1= HxPortText(self, "keep_id_1", "IDs to Keep for seed 1")
       
        self.keep_id_2= HxPortText(self, "keep_id_2","IDs to Keep for seed 1")
        

        
        self.input_seed2 = HxConnection(self, "input_seed2", "Input seed 2")
        self.input_seed2.valid_types = ('HxUniformLabelField3')
        
        self.keep_id_1= HxPortText(self, "keep_id_1", "IDs to Keep for seed 1")
       
        self.keep_id_2= HxPortText(self, "keep_id_2","IDs to Keep for seed 1")
        

        self.do_it = HxPortDoIt(self, "apply", "Apply")

        _hx_core._tcl_interp('stopLogInhibitor')





    def load_files(self, file_paths):
        """Loading data into avizo.

        Args:-3.
            file_paths (_type_): A list of file paths=.0

        Returns:
            _type_: _description_
        """

        import time

        with hx_progress.progress(len(file_paths), f"Loading {len(file_paths)} file") as progress:
            for file_path in file_paths:
                progress.increase_progress_step()
                self.files.append(hx_project.load(file_path))
                
                if progress.interrupted:
                    break
            
            # for i in range(10):
            #     progress.increase_progress_step()
            #     time.sleep(1) # We just wait instead of doing some interesting computations.
            #     print("interrupted by user: {progress.interrupted} step: {progress.current_step} progress: {progress.value:.0%}".format(progress=progress))
            #     if progress.interrupted:
            #         break
        
        # return files

    def to_label(self):
        ### Code for convert seeds (images) into 8-bits label
        ### So Avizo can visualise them as labels.

        for file in self.files:
            
            convert = hx_project.create("HxCastField")
            self.converts.append(convert)
            time.sleep(2)

            convert.ports.data.connect(file)
            convert.fire()
            # convert.ports.outputType.menus[0].options[0] = '8-bit label'
            # 7 is '8-bit label'
            convert.ports.outputType.menus[0].selected = 7
            convert.fire()
            # if convert.ports.outputType.menus[0].selected ==7:
                # hx_message.info("Computation was a success !")
                

            set_prev = set(hx_project.objects.keys())
            convert.execute()

            name = list((set(hx_project.objects.keys()).difference(set_prev)))[0]
            print(name)
            self.segs.append(hx_project.get(name))

        ## Remove all the original images and covnert data type
        del_all_objs(self.files)
        del_all_objs(self.converts)    
        
    def view_obj_vol_ren(self, obj):
        """Visualise an object using volume rendering

        Args:
            obj (_type_): object's variable. or hx_project.get(<name of the variable>)
            
        Returns:
            _type_: HxVolumeRender2
        """
        try:
            hx_project.remove(self.vol_ren_set)
            hx_project.remove(self.vol_ren)
        except:
            pass
        self.vol_ren_set  = hx_project.create("HxVolumeRenderingSettings")
        self.vol_ren = hx_project.create("HxVolumeRender2")
        self.vol_ren.ports.volumeRenderingSettings.connect(self.vol_ren_set)
        self.vol_ren_set.ports.data.connect(obj)
        self.vol_ren_set.fire()            
   
    def view_obj_mesh(self, obj):
        """Visualise an object using volume rendering

        Args:
            obj (_type_): object's variable. or hx_project.get(<name of the variable>)
            
        Returns:
            _type_: HxVolumeRender2
        """
        try:
            hx_project.remove(self.GMC)
            self.surf_view.viewer_mask = 0
            hx_project.remove(self.surf_view)
            hx_project.remove(self.surf)
        except:
            pass
        
        
        self.GMC  = hx_project.create("HxGMC")
        self.GMC.ports.smoothingExtent.value = 1
        self.GMC.ports.data.connect(obj)
        
        set_prev = set(hx_project.objects.keys())
        self.GMC.execute()
    
        surf_name = list((set(hx_project.objects.keys()).difference(set_prev)))[0]
        print(surf_name)
        self.surf = hx_project.get(surf_name)
        self.surf_view = hx_project.create("HxDisplaySurface")
        
        self.surf_view.ports.data.connect(self.surf)
        # GMC.fire()
        self.surf_view.execute()        
    
    def toggle_one_img(self,visible):
        self.input_seed.visible = visible    
        
        self.src_ids.visible = visible    
        
        self.dst_ids.visible = visible    
    
    def toggle_two_img(self, visible):
        
        self.input_seed1.visible = visible    
        self.input_seed2.visible = visible    
        self.keep_id_1.visible = visible    
       
        self.keep_id_2.visible = visible    
    
    def update(self):

        
        if self.functions.selected ==0:
            self.toggle_one_img(True)
            self.toggle_two_img(False)
        elif self.functions.selected ==1:
            self.toggle_one_img(False)
            self.toggle_two_img(True)
        
        # if self.load_mode.selected == 0:
        #     self.input_multi_files.enabled = False
        #     self.input_dir.enabled = True
        # elif self.load_mode.selected == 1:
        #     self.input_multi_files.enabled = True
        #     self.input_dir.enabled = False
        
    
        pass
    
    
    def compute(self):
        # Does the user press the 'Apply' button?
        if not self.do_it.was_hit:
            return


        if self.functions.selected ==0:
            str_dst_ids = self.dst_ids.text
            str_src_ids = self.src_ids.text
            
            if not is_valid_src_ids_list(str_src_ids):
                print(f"Src_ids is not valid, please input one more multi class id and separate by comma\n" \
                    "like \'1\' or \'1, 2, 3, 4\'  ")
            
            if not is_valid_dst_ids_list(str_dst_ids):
                print(f"dst_ids is not valid, please input one class id")
                
            dst_ids = string_to_integer_list(str_dst_ids)
            src_ids = string_to_integer_list(str_src_ids)
            
            print(f"Merging ids: {src_ids}")
            print(f"Target id: {dst_ids}")
            
            # Is there an input data connected?
            if self.input_seed1.source() is None:
                "Please select the input"
                return
            
            input_1 = self.input_seed.source()
            np_input_1 = input_1.get_array()
            
            print(type(input_1))
        
            output = replace_values_in_array(np_input_1, src_ids, dst_ids)

            result = hx_project.create('HxUniformLabelField3')
            result.name = input_1.name + "Merged.Segmentation"
            result.bounding_box = input_1.bounding_box
            result.set_array(np.array(output, dtype = np.uint8))
        elif self.functions.selected ==1:
            str_keep_id_1 = self.keep_id_1.text
            str_keep_id_2 = self.keep_id_2.text
            
            if not (is_valid_src_ids_list(str_keep_id_1) or is_valid_src_ids_list(str_keep_id_2)):
                print(f"Id format is not valid, please input one more multi class id and separate by comma\n" \
                    "like \'1\' or \'1, 2, 3, 4\'  ")
                            
            keep_id_1 = string_to_integer_list(str_keep_id_1)
            keep_id_2 = string_to_integer_list(str_keep_id_2)
            
            print(f"Ids to keep for seed 1: {keep_id_1}")
            print(f"Ids to keep for seed 2: {keep_id_2}")
            
            # Is there an input data connected?
            if (self.input_seed1.source() is None) or (self.input_seed2.source() is None):
                print("Please select the input")
                return
            
            input_1 = self.input_seed1.source()
            np_input_1 = input_1.get_array()
            
            input_2 = self.input_seed2.source()
            np_input_2 = input_2.get_array()
            
            output = merge_masks_with_filter(np_input_1, np_input_2,keep_id_1,keep_id_2)


            result = hx_project.create('HxUniformLabelField3')
            result.name = input_1.name + "two_img_merge.Segmentation"
            result.bounding_box = input_1.bounding_box
            result.set_array(np.array(output, dtype = np.uint8))
        # if self.functions.selected ==0:
            
        #     if self.load_mode.selected == 0:
        #         print(f"reading files from: {self.input_dir.filenames}")
                
        #         if not is_valid_path(self.input_dir.filenames):
        #             print("File path not valid, please reinput")
        #             hx_message.info("File path not valid, please reinput")
        #         else:  
        #             file_paths = glob.glob(os.path.join(self.input_dir.filenames,"*.tif"))
        #             hx_message.info(f"reading {len(file_paths)} files from: {self.input_dir.filenames}")
        #     elif self.load_mode.selected == 1:
        #         file_paths= self.input_multi_files.filenames
                
            
        #     self.load_files(file_paths)
        #     self.to_label()
        # elif self.functions.selected ==1:
        #     if self.inputL.source() is None:
        #         hx_message.info(f"Please select an input label")
        #         return
        #     labeldata = self.inputL.source()
    
        #     if self.vis_mode.selected == 0:
        #         self.view_obj_vol_ren(labeldata)
        #     elif self.vis_mode.selected == 1:
        #         self.view_obj_mesh(labeldata)
    
