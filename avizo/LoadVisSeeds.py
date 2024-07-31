##help codes for avizo console
# addon = hx_project.objects.get("read_addon")
# addon.portnames
import glob
import os
from pathlib import Path
import time

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
class LoadVisSeeds(PyScriptObject):
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

        # This port will define number of iterations.
        # self.input_dir= HxPortTextEdit(self, "Path", "Folder of input segs")
        # self.input_dir.row_qty = 1
        
        self.functions = HxPortRadioBox(self,"functions", "Choose Load Files or visualsation")
        self.functions.radio_boxes = [
            HxPortRadioBox.RadioBox(label="Load Files"),
        HxPortRadioBox.RadioBox(label="visualsation")]
        self.functions.selected = 0
        
        self.load_mode = HxPortRadioBox(self,"load_mode", "Choose folder or multi files")
        self.load_mode.radio_boxes = [
            HxPortRadioBox.RadioBox(label="Folder"),
            HxPortRadioBox.RadioBox(label="Multi files")]
        self.load_mode.selected = 0
        
        self.input_dir= HxPortFilename(self, "input_dir", "Folder of input segs")
        self.input_dir.mode = HxPortFilename.LOAD_DIRECTORY
        
        
        self.input_multi_files= HxPortFilename(self, "input_multi_files", "Folder of input segs")
        self.input_multi_files.mode = HxPortFilename.MULTI_FILE
        self.input_multi_files.enabled = False
        

        self.vis_mode = HxPortRadioBox(self,"vis_mode", "How to visualise")
        self.vis_mode.radio_boxes = [
            HxPortRadioBox.RadioBox(label="Volume"),
            HxPortRadioBox.RadioBox(label="Mesh")]
        self.load_mode.selected = 0

        
        self.inputL = HxConnection(self, "inputL", " Label to visualise")
        self.inputL.valid_types = ('HxUniformLabelField3')
        # self.input_dir.row_qty = 1
        
        # self.iters.clamp_range = (1, 1000)
        # self.iters.value = 7

        # # This port will define initial threshold.
        # self.threshinit = HxPortIntSlider(self, "threshinit", "Initial Threshold")
        # self.threshinit.clamp_range = (0, 100000)
        # self.threshinit.value = 34000

        # # This port will define target threshold.
        # self.threshfin = HxPortIntSlider(self, "threshfin", "Target Threshold")
        # self.threshfin.clamp_range = (0, 100000)
        # self.threshfin.value = 21500


        # # Seed Dilation
        # self.sradio_boxes = HxPortRadioBox(self,"SD", "Seed Dilation")
        # self.sradio_boxes.radio_boxes = [HxPortRadioBox.RadioBox(label="On"),
        # HxPortRadioBox.RadioBox(label="Off")]
        # self.sradio_boxes.selected = 0

        # # Label preservation
        # self.lpradio_boxes = HxPortRadioBox(self,"LP", "Label Preservation")
        # self.lpradio_boxes.radio_boxes = [HxPortRadioBox.RadioBox(label="On"),
        # HxPortRadioBox.RadioBox(label="Off")]
        # self.lpradio_boxes.selected = 1

        # # Save Seed
        # self.ssradio_boxes = HxPortRadioBox(self,"SS", "Save Seed")
        # self.ssradio_boxes.radio_boxes = [HxPortRadioBox.RadioBox(label="On"),
        # HxPortRadioBox.RadioBox(label="Off")]
        # self.ssradio_boxes.selected = 1


        # Classic 'Apply' button:
        self.do_it = HxPortDoIt(self, "apply", "Apply")

        _hx_core._tcl_interp('stopLogInhibitor')





    def load_files(self, file_paths):
        """Loading data into avizo.

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
    
    def toggle_load(self,visible):
        self.input_multi_files.visible = visible
        self.input_dir.visible = visible
        self.load_mode.visible = visible
    
    def toggle_vis(self, visible):
        self.inputL.visible = visible    
        self.vis_mode.visible = visible   
    
    def update(self):
        # if not self.inputL.is_new:
        #     return
        # if self.input.source() is None:
        
        if self.functions.selected ==0:
            self.toggle_load(True)
            self.toggle_vis(False)
        elif self.functions.selected ==1:
            self.toggle_load(False)
            self.toggle_vis(True)
        
        if self.load_mode.selected == 0:
            self.input_multi_files.enabled = False
            self.input_dir.enabled = True
        elif self.load_mode.selected == 1:
            self.input_multi_files.enabled = True
            self.input_dir.enabled = False
        
        
        # if len(self.segs )==0:
        #     self.inputL.visible = False
        # else:
        #     self.inputL.visible = True
        #     self.iters.visible = True
        #     self.threshinit.visible = True
        #     self.threshfin.visible = True
        #     self.sradio_boxes.visible = True
        #     self.ssradio_boxes.visible = True
        #     self.lpradio_boxes.visible = False
        # elif self.inputL.source is not None:
        #     self.iters.visible = True
        #     self.threshinit.visible = True
        #     self.threshfin.visible = True
        #     self.sradio_boxes.visible = True
        #     self.ssradio_boxes.visible = True
        #     self.lpradio_boxes.visible = True
        pass
    
    
    def compute(self):
        # Does the user press the 'Apply' button?
        if not self.do_it.was_hit:
            return

        # Is there an input data connected?
        # if self.input.source() is None:
        #     return
        
        if self.functions.selected ==0:
            
            if self.load_mode.selected == 0:
                print(f"reading files from: {self.input_dir.filenames}")
                
                if not is_valid_path(self.input_dir.filenames):
                    print("File path not valid, please reinput")
                    hx_message.info("File path not valid, please reinput")
                else:  
                    file_paths = glob.glob(os.path.join(self.input_dir.filenames,"*.tif"))
                    hx_message.info(f"reading {len(file_paths)} files from: {self.input_dir.filenames}")
            elif self.load_mode.selected == 1:
                file_paths= self.input_multi_files.filenames
                print(file_paths, type(file_paths))
                if isinstance(file_paths, str):
                    file_paths = [file_paths]
                
            
            self.load_files(file_paths)
            self.to_label()
            
            
  
            
            # self.files = []
            # self.converts = []
            
            # self.input_dir.filenames = ""
            # self.input_multi_files.filenames = ""
            
        elif self.functions.selected ==1:
            if self.inputL.source() is None:
                hx_message.info(f"Please select an input label")
                return
            labeldata = self.inputL.source()
    
            if self.vis_mode.selected == 0:
                self.view_obj_vol_ren(labeldata)
            elif self.vis_mode.selected == 1:
                self.view_obj_mesh(labeldata)
    

    
        # # Retrieve the input data.
        # input = self.input.source()
        # labeldata = self.inputL.source()
        # # Check if the output field exists and create it
        # if isinstance(self.results[0], HxRegScalarField3) is False:
        #     result = hx_project.create('HxUniformLabelField3')
        #     result.name = input.name + ".Segmentation"
        # else:
        #     result = self.results[0]

        # import numpy as np
        # from scipy import ndimage as ndi
        # import warnings
        # from skimage.morphology import ball
        # import gc


        # def get_largest(label, segments):
        #     labels, _ = ndi.label(label)
        #     assert (labels.max() != 0)
        #     number = 0
        #     try:
        #         bincount = np.bincount(labels.flat)[1:]
        #         bincount_sorted = np.sort(bincount)[::-1]
        #         largest = labels - labels
        #         m = 0
        #         for i in range(segments):
        #             index = int(np.where(bincount == bincount_sorted[i])[0][m]) + 1
        #             ilargest = labels == index
        #             largest += np.where(ilargest, i + 1, 0)
        #         if i == segments - 1:
        #             number = segments
        #     except:
        #         warnings.warn(f"Number of segments should be reduced to {i}")
        #         if number == 0:
        #             number = i
        #     return largest, number

        # def grow(labels, number):
        #     grownlabels = np.copy(labels)
        #     for i in range(number):
        #         filtered = np.where(labels == i + 1, 1, 0)
        #         grown = ndi.binary_dilation(np.copy(filtered), structure=ball(2)).astype(np.uint16)
        #         grownlabels = np.where(np.copy(grown), i + 1, np.copy(grownlabels))
        #         del grown
        #         del filtered
        #     return grownlabels

        # def bbox2_3D(img):
        #     r = np.any(img, axis=(1, 2))
        #     c = np.any(img, axis=(0, 2))
        #     z = np.any(img, axis=(0, 1))

        #     rmin, rmax = np.where(r)[0][[0, -1]]
        #     cmin, cmax = np.where(c)[0][[0, -1]]
        #     zmin, zmax = np.where(z)[0][[0, -1]]

        #     return rmin, rmax, cmin, cmax, zmin, zmax

        # def segmentation(volume_array, initial_threshold, target_threshold, segments, iterations, label=False,
        #                  label_preserve=False, seed_dilation=False):

        #     if type(label) == bool:
        #         volume_label = volume_array > initial_threshold
        #     else:
        #         volume_label = label

        #     if label_preserve == False:
        #         seed, number = get_largest(volume_label, segments)
        #     else:
        #         seed = volume_label
        #         number = segments

        #     if seed_dilation == True:
        #         formed_seed = grow(seed, number)
        #     else:
        #         formed_seed = seed

        #     labeled_volume = np.copy(formed_seed)
        #     with hx_progress.progress(iters * number, "Refining") as progress:
        #         for i in range(iterations + 1):
        #             volume_label = volume_array > initial_threshold - (
        #                         i * (initial_threshold - target_threshold) / iterations)
        #             volume_label = np.where(labeled_volume != 0, False, volume_label)
        #             if progress.interrupted:
        #                 break
        #             for j in range(number):
        #                 hx_progress.set_text(f"Refining -- Iter:{i} -- Label:{j}")
        #                 try:
        #                     rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(labeled_volume == j + 1)
        #                 except:
        #                     rmin, rmax, cmin, cmax, zmin, zmax = -1, 1000000, -1, 1000000, -1, 1000000
        #                 maximum = labeled_volume.shape
        #                 rmin = max(0, rmin - int((rmax - rmin) * 0.1))
        #                 rmax = min(int((rmax - rmin) * 0.1) + rmax, maximum[0])
        #                 cmin = max(0, cmin - int((cmax - cmin) * 0.1))
        #                 cmax = min(int((cmax - cmin) * 0.1) + cmax, maximum[1])
        #                 zmin = max(0, zmin - int((zmax - zmin) * 0.1))
        #                 zmax = min(int((zmax - zmin) * 0.1) + zmax, maximum[2])
        #                 temp_label = np.copy(volume_label)
        #                 reduced_labeled_volume = labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax]
        #                 temp_label[rmin:rmax, cmin:cmax, zmin:zmax] = np.copy(volume_label)[rmin:rmax, cmin:cmax,
        #                                                               zmin:zmax] + (
        #                                                                       reduced_labeled_volume == j + 1)
        #                 pos = np.where(reduced_labeled_volume == j + 1)
        #                 labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(reduced_labeled_volume == j + 1, 0,
        #                                                                            reduced_labeled_volume)
        #                 labeled_temp, _ = ndi.label(np.copy(temp_label[rmin:rmax, cmin:cmax, zmin:zmax]))
        #                 try:
        #                     index = int(labeled_temp[pos[0][0], pos[1][0], pos[2][0]])
        #                 except:
        #                     index = 1
        #                 try:
        #                     relabelled = np.copy(labeled_temp) == index
        #                     labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(np.copy(relabelled), j + 1,
        #                                                                                labeled_volume[rmin:rmax, cmin:cmax,
        #                                                                                zmin:zmax])
        #                     del temp_label
        #                     del pos
        #                     del labeled_temp
        #                     del relabelled
        #                     gc.collect()
        #                     progress.current_step = i*number+j
        #                     if progress.interrupted:
        #                         break
        #                 except:
        #                     print(f"missing {j}")
        #             # step = hx_project.create('HxUniformLabelField3')
        #             # step.name = input.name + f".Step{i}"
        #             # step.set_array(np.array(labeled_volume, dtype=np.ushort))
        #             # step.bounding_box = input.bounding_box
        #     return labeled_volume, formed_seed

        # # Retrieve the values:
        # number = self.n.value
        # iters = self.iters.value
        # initial_threshold = self.threshinit.value
        # target_threshold = self.threshfin.value
        # dilation = self.sradio_boxes.selected
        # preservation = self.lpradio_boxes.selected
        # saveseed = self.ssradio_boxes.selected

        # if dilation == 0:
        #     dilation = True
        # else:
        #     dilation = False
        # if preservation == 0:
        #     preservation = True
        # else:
        #     preservation = False
        # if saveseed == 0:
        #     saveseed = True
        # else:
        #     saveseed = False

        # # Compute our output array:
        # volume_array = np.array(input.get_array(), dtype = np.ushort)
        # if labeldata:
        #     labeled = np.array(labeldata.get_array(), dtype=np.ushort)
        # else:
        #     labeled = False

        # labeled_volume, formed_seed = segmentation(volume_array,initial_threshold,target_threshold,number,iters,label = labeled,label_preserve=preservation,seed_dilation=dilation)

        # # convert
        # mConvert = hx_project
        # result.set_array(np.array(labeled_volume, dtype = np.ushort))
        # if saveseed:
        #     seed = hx_project.create('HxUniformLabelField3')
        #     seed.name = input.name + ".Seed"
        #     seed.set_array(np.array(formed_seed, dtype=np.ushort))
        #     seed.bounding_box = input.bounding_box
        # # Output bounding-box is the same as input bbox.
        # result.bounding_box = input.bounding_box




        # # Set as current result.
        # self.results[0] = result

