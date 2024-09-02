##help codes for avizo console
# addon = hx_project.objects.get("read_addon")
# addon.portnames
import glob
import os
from pathlib import Path
import time
import numpy as np
from tifffile import imread

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
        

import re

def extract_number(data_string):
    # Regular expression to find the number in the string
    match = re.search(r'[\d\.]+', data_string)
    if match:
        return float(match.group())
    return None

def extract_bit_depth(data_string):
    # Regular expression to find the word ending with '-bit' and capture the preceding number
    match = re.search(r'(\d+)-bit', data_string, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def process_voxel_dimensions(dim_string):
    # Split the string by 'x' and remove any leading/trailing whitespace
    dimensions = dim_string.split('x')
    
    # Convert each part to a float
    voxel_x = float(dimensions[0].strip())
    voxel_y = float(dimensions[1].strip())
    voxel_z = float(dimensions[2].strip())
    
    return voxel_x, voxel_y, voxel_z


def print_material_cls_mapping(input):
    input_seed = input.source()
    
    data_info_text = input_seed.ports.DataInfo.text
    
    if "window" not in data_info_text:
        hx_message.info("Material IDs match class IDs")   

    else:
        mats = eval(str(input_seed.parameters['Materials']))
        
        keys = list(mats.keys())
        arr_ids = np.unique(input_seed.get_array())
        output_str = "Material and Segmentation class mapping\n"
        
        for key, arr_id in zip(keys, arr_ids):
            output_str+=f"{key}:    Seg class {arr_id}\n"
        # for key, value in mats.items():
        #     print(key, value['Id'])

        hx_message.info(output_str)   
    

def test_progress_bar():
    with hx_progress.progress(10, f"Loading 10 for progress bar") as progress:
        for i in range(10):
            progress.increase_progress_step()
            time.sleep(1) # We just wait instead of doing some interesting computations.
            print("interrupted by user: {progress.interrupted} step: {progress.current_step} progress: {progress.value:.0%}".format(progress=progress))
            if progress.interrupted:
                break
class PreProcess(PyScriptObject):
    def __init__(self):
        
        self.segs = []
        self.converts = []
        self.files = []
        import _hx_core
        _hx_core._tcl_interp('startLogInhibitor')

        self.input_data = HxConnection(self, "input_data", " Data")
        self.input_data.valid_types = ('HxRegScalarField3')
        
        self.min_coords = HxPortIntTextN(self,"min_coords", "Crop Min")
        self.min_coords.texts = [HxPortIntTextN.IntText(label="X"),
                             HxPortIntTextN.IntText(label="Y"),
                             HxPortIntTextN.IntText(label="Z")]
        
        self.max_coords = HxPortIntTextN(self,"max_coords", "Crop Max")
        self.max_coords.texts = [HxPortIntTextN.IntText(label="X"),
                             HxPortIntTextN.IntText(label="Y"),
                             HxPortIntTextN.IntText(label="Z")]
        
        self.ren_thres = HxPortIntTextN(self,"ren_thres", "Volume rendering thresholds")
        self.ren_thres.texts = [HxPortIntTextN.IntText(label="Min",
                                                       value=-1, clamp_range=(-1,65535)),
                             HxPortIntTextN.IntText(label="Max",
                                                    value=-1, clamp_range=(-1,65535))]
                
        for text in self.min_coords.texts:
            text.value = 0
            text.clamp_range = (0,10000)
        for text in self.max_coords.texts:
            text.value = 0
            text.clamp_range = (0,10000)
            
            
        self.is_size_check = HxPortToggleList(self,"is_size_check", "Size limit")
        self.is_size_check.toggles[0] = HxPortToggleList.Toggle(label="True", checked=HxPortToggleList.Toggle.CHECKED)
        
        self.size_check = HxPortFloatTextN(self,"size_check", "Parameters for resize")
        self.size_check.texts = [HxPortFloatTextN.FloatText(label="Size limit (MB)",
                                                            value=1500, clamp_range=(0.0, 20000.0)),
                                 HxPortFloatTextN.FloatText(label="Resample Scale", value = 2),
                                 ]
        
        
        self.swap_axes = HxPortRadioBox(self,"swap_axes", "Choose axes to swap")
        self.swap_axes.radio_boxes = [
            HxPortRadioBox.RadioBox(label="None"),
            HxPortRadioBox.RadioBox(label="X and Y"),
            HxPortRadioBox.RadioBox(label="X and Z"),
            HxPortRadioBox.RadioBox(label="Y and Z"),
            ]
        self.swap_axes.selected = 0

        # self.size_check.texts[0].value = 1500
        # self.size_check.texts[1].value = 2
        # self.functions = HxPortRadioBox(self,"functions", "Choose Load Files or visualsation")
        # self.functions.radio_boxes = [
        #     HxPortRadioBox.RadioBox(label="Load Files"),
        # HxPortRadioBox.RadioBox(label="visualsation")]
        # self.functions.selected = 0
        
        # self.load_mode = HxPortRadioBox(self,"load_mode", "Choose folder or multi files")
        # self.load_mode.radio_boxes = [
        #     HxPortRadioBox.RadioBox(label="Folder"),
        #     HxPortRadioBox.RadioBox(label="Multi files")]
        # self.load_mode.selected = 0
        

        
        # self.input_dir= HxPortFilename(self, "input_dir", "Folder of input segs")
        # self.input_dir.mode = HxPortFilename.LOAD_DIRECTORY
        
        
        # self.input_multi_files= HxPortFilename(self, "input_multi_files", "Folder of input segs")
        # self.input_multi_files.mode = HxPortFilename.MULTI_FILE
        # self.input_multi_files.enabled = False
        

        # self.vis_mode = HxPortRadioBox(self,"vis_mode", "How to visualise")
        # self.vis_mode.radio_boxes = [
        #     HxPortRadioBox.RadioBox(label="Volume"),
        #     HxPortRadioBox.RadioBox(label="Mesh"),
        #     HxPortRadioBox.RadioBox(label="View Mat and Seg class")]
        # self.vis_mode.selected = 0




        
        # self.inputL = HxConnection(self, "inputL", " Label to visualise")
        # self.inputL.valid_types = ('HxUniformLabelField3')



        # Classic 'Apply' button:
        self.do_it = HxPortDoIt(self, "apply", "Apply")

        _hx_core._tcl_interp('stopLogInhibitor')





    def load_files(self, file_paths):
        """Loading data into avizo, using hx_project.load()

        Args:-3.
            file_paths (_type_): A list of file paths=.0

        Returns:
            _type_: _description_
        """
        self.files = []
        self.converts = []

        import time
        print(f"Loading {len(file_paths)} file")
        with hx_progress.progress(len(file_paths), f"Loading {len(file_paths)} file") as progress:
            for file_path in file_paths:
                progress.increase_progress_step()
                result = hx_project.load(file_path)
                if isinstance(result, list):
                    for r in result:
                        if 'HxRegScalarField3' in str((type(r))):
                            self.files.append(r)
                        else:
                            hx_project.remove(r)
                else:
                    self.files.append(result)
                
                
                
                if progress.interrupted:
                    break
            

    def load_data_tifffile(self, file_paths):
        """Loading data into avizo, using hx_project.load()

        Args:-3.
            file_paths (_type_): A list of file paths=.0

        Returns:
            _type_: _description_
        """
        self.files = []
        self.converts = []

        import time
        print(f"Loading {len(file_paths)} file")
        with hx_progress.progress(len(file_paths), f"Loading {len(file_paths)} file") as progress:
            for file_path in file_paths:
                progress.increase_progress_step()
                img = imread(file_path)
                # Extract the dimensions
                z, height, width = img.shape  
                bbox = ((0.0, 0.0, 0.0), (width, height, z))
                
                img_avizo = hx_project.create('HxUniformLabelField3')
                img_avizo.name = os.path.basename(file_path)
                
                img_avizo.bounding_box = bbox
                img_avizo.set_array(img)
                
                self.files.append(img_avizo)
                
                if progress.interrupted:
                    break


    def to_label(self,  is_reorder):
        ### Code for convert images (default type after loading in Avizo) into 8-bits label
        ### So Avizo can visualise them as labels.
        print(f"Reorder status: {is_reorder}")

        if is_reorder==1:
            reorder = hx_project.create("reorder_labels")
        
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

            convert_result = convert.results[0]
            if is_reorder!=1:
                self.segs.append(convert_result)
            elif is_reorder==1:
                reorder.ports.inputLabelImage.connect(convert_result)
                reorder.fire()
                reorder.execute()
                reorder_result = reorder.results[0]
                
                self.segs.append(reorder_result)
                
                reorder.results[0] = None
                reorder.ports.inputLabelImage.disconnect()
                hx_project.remove(convert_result)
                
            
            # name = list((set(hx_project.objects.keys()).difference(set_prev)))[0]
            # print(name)
            # self.segs.append(hx_project.get(name))

        ## Remove all the original images and covnert data type
        del_all_objs(self.files)
        del_all_objs(self.converts)
        if is_reorder==1:
            hx_project.remove(reorder)
        
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
        
        threshold_range = list(self.vol_ren.ports.colormap.range)
        
        min_thre, max_thre = self.ren_thres.texts[0].value, self.ren_thres.texts[1].value
        
        if min_thre == -1:
            min_thre = threshold_range[0]
        if max_thre == -1:
            max_thre = threshold_range[1]
        
        self.vol_ren.ports.colormap.range= [min_thre,max_thre]
        self.vol_ren.fire()
   
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
    
    def toggle_load(self,visible):
        self.input_multi_files.visible = visible
        self.input_dir.visible = visible
        self.load_mode.visible = visible
        self.is_reorder.visible = visible
    
    def toggle_vis(self, visible):
        self.inputL.visible = visible    
        self.vis_mode.visible = visible   
    
    def toggle_size_check(self, visible):
        self.size_check.visible = visible
    
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
        # if not self.inputL.is_new:
        #     return
        # if self.input.source() is None:
        self.addon_clean()
    
        self.toggle_size_check(self.is_size_check.toggles[0].checked)
        
    
    
    def compute(self):
        # Does the user press the 'Apply' button?
        if not self.do_it.was_hit:
            return

        # Is there an input data connected?
        if self.input_data.source() is None:
            return
        

        # self.min_coords = HxPortIntTextN(self,"min_coords", "Crop Min")
        # self.min_coords.texts = [HxPortIntTextN.IntText(label="X"),
        #                      HxPortIntTextN.IntText(label="Y"),
        #                      HxPortIntTextN.IntText(label="Z")]
        
        # self.max_coords = HxPortIntTextN(self,"max_coords", "Crop Max")
        # self.max_coords.texts = [HxPortIntTextN.IntText(label="X"),
        #                      HxPortIntTextN.IntText(label="Y"),
        #                      HxPortIntTextN.IntText(label="Z")]
        

        input = self.input_data.source()
        np_input = input.get_array()
        
        # Get the dimensions
        x_dim, y_dim, z_dim = np_input.shape

        # Print the dimensions
        print(f"Dimensions of the 3D image: x = {x_dim}, y = {y_dim}, z = {z_dim}")
        
        min_x,min_y,min_z = self.min_coords.texts[0].value,self.min_coords.texts[1].value,self.min_coords.texts[2].value
        
        max_x,max_y,max_z = self.max_coords.texts[0].value,self.max_coords.texts[1].value,self.max_coords.texts[2].value
        

        cropped = np_input[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
        
        ##If the swap axes is needed
        if self.swap_axes.selected == 1:
            cropped = np.swapaxes(cropped, 0, 1)
        elif self.swap_axes.selected == 2:
            cropped = np.swapaxes(cropped, 0, 2)
        elif self.swap_axes.selected == 3:
            cropped = np.swapaxes(cropped, 1, 2)
        
        result = hx_project.create('HxUniformScalarField3')
        result.name = input.name + "_cropped"
        result.bounding_box = ((0.0, 0.0, 0.0), tuple([s-1 for s in cropped.shape]))
        result.set_array(cropped)
        
        self.view_obj_vol_ren(result)
        print(f"min_x,min_y,min_z:{[min_x,min_y,min_z]}")
        print(f"max_x,max_y,max_z:{[max_x,max_y,max_z]}")
        
        result.selected = True
        print(result.name, result.portnames)
        
        

        
        
        # #### Resizing the image size
        if self.is_size_check.toggles[0].checked:
            size_str = result.ports.MemorySize.text
            data_info = result.ports.DataInfo.text
            voxel_str = result.ports.VoxelSize.text

            voxel_x, voxel_y, voxel_z = process_voxel_dimensions(voxel_str)


            size = extract_number(size_str)
            bit = extract_bit_depth(data_info)

            size_limit = self.size_check.texts[0].value
            resize_scale = self.size_check.texts[1].value

            if size > size_limit:
                
                resample = hx_project.create("HxResample")
                
                if bit ==8:
                    print("Reducing size by resampling")
                    
                    
                else:
                    print("Reducing size by (1) Resampling and (2) Convert to 8 bit")
                    
                    
                    convert = hx_project.create("HxCastField")
                    convert.ports.data.connect(result)
                    convert.fire()
                    convert.execute()
                    
                    convert_result = convert.results[0]
                    
                    convert.results[0] = None

                resample.ports.data.connect(result)
                resample.fire()

                resample.ports.mode.selected = 1
                resample.fire()
                voxelSize_set = resample.ports.voxelSize
                
                voxelSize_set.texts[0].value = voxel_x * resize_scale
                voxelSize_set.texts[1].value = voxel_y * resize_scale
                voxelSize_set.texts[2].value = voxel_z * resize_scale
                

                resample.execute()
                
                resample_result = resample.results[0]
                
                resample.results[0] = None
        
       