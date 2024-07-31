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
class LoadVisFiles(PyScriptObject):
    def __init__(self):
        
        self.segs = []
        self.converts = []
        self.files = []
        import _hx_core
        _hx_core._tcl_interp('startLogInhibitor')


        
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
        ### Code for convert images (default type after loading in Avizo) into 8-bits label
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
    

    

