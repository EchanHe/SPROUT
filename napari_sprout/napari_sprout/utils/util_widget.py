import os
from qtpy.QtWidgets import (
    QGroupBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QCheckBox, QFileDialog
)

def create_output_folder_row(default_folder=None):

    layout = QHBoxLayout()
    label = QLabel("Output Folder:")
    folder_edit = QLineEdit()
    browse_btn = QPushButton("Browse")

    # set default folder if provided, otherwise use current working directory
    if default_folder is None:
        default_folder = os.path.join(os.getcwd(), "sprout_temp")
    os.makedirs(default_folder, exist_ok=True)
    folder_edit.setText(default_folder)

    def browse():
        folder = QFileDialog.getExistingDirectory(None, "Select Output Folder")
        if folder:
            folder_edit.setText(folder)

    browse_btn.clicked.connect(browse)

    layout.addWidget(label)
    layout.addWidget(folder_edit)
    layout.addWidget(browse_btn)

    return layout, folder_edit



class GrowOptionalParamGroupBox(QGroupBox):
    def __init__(self, default_output=None):
        super().__init__("Advanced Grow Parameters")
        # self.setCheckable(True)
        # self.setChecked(False)

        layout = QFormLayout()

        # Save every n iterations
        self.save_iter_checkbox = QCheckBox("Save every N iterations")
        self.save_iter_checkbox.stateChanged.connect(self._toggle_save_iters)
        self.save_every_n_spin = QSpinBox()
        self.save_every_n_spin.setRange(0, 100)
        self.save_every_n_spin.setEnabled(False)
        layout.addRow(self.save_iter_checkbox, self.save_every_n_spin)

        # Grow to end
        self.grow_to_end_checkbox = QCheckBox("Grow to end")
        layout.addRow(self.grow_to_end_checkbox)

        # Sort seed ids
        self.sort_checkbox = QCheckBox("Sort the result by size")
        layout.addRow(self.sort_checkbox)

        # Grow specific IDs
        self.id_list_line = QLineEdit()
        self.id_list_line.setPlaceholderText("e.g. 1,3,5")
        layout.addRow(QLabel("IDs to Grow (optional):"), self.id_list_line)

        self.early_stop_checkbox = QCheckBox("Early stop if no growth")
        self.early_stop_checkbox.setToolTip("Stop growing if no growth occurs for a specified number of iterations.")
        self.early_stop_checkbox.setChecked(True)
        self.early_stop_checkbox.stateChanged.connect(self._toggle_no_growth_spin)
        layout.addRow(self.early_stop_checkbox)
        # Max no-growth iters
        self.no_growth_spin = QSpinBox()
        self.no_growth_spin.setRange(1, 100)
        self.no_growth_spin.setValue(3)
        self.no_growth_spin.setEnabled(True)
        layout.addRow(QLabel("Max No-Growth Iterations:"), self.no_growth_spin)

        # Min growth size
        self.min_growth_spin = QSpinBox()
        self.min_growth_spin.setRange(0, 1000000)
        self.min_growth_spin.setValue(50)
        self.min_growth_spin.setEnabled(True)
        layout.addRow(QLabel("Minimum Growth Size:"), self.min_growth_spin)



        self.setLayout(layout)

    def _toggle_no_growth_spin(self):
        self.no_growth_spin.setEnabled(self.early_stop_checkbox.isChecked())
        # if not self.early_stop_checkbox.isChecked():
        #     self.no_growth_spin.setValue(3)
        self.min_growth_spin.setEnabled(self.early_stop_checkbox.isChecked())
    
    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_line.setText(folder)

    def _toggle_save_iters(self):
        self.save_every_n_spin.setEnabled(self.save_iter_checkbox.isChecked())

    def get_params(self):
        to_grow_text = self.id_list_line.text().strip()
        to_grow_text.replace(" ", "")  # remove any spaces

        try:
            to_grow_ids = [int(x) for x in to_grow_text.split(",") if x.strip().isdigit()]
        except:
            to_grow_ids = None
            # show error in napari message box
            from napari.utils.notifications import show_error
            show_error("Invalid IDs format. Please enter a comma-separated list of integers.")
       
        # if to_grow_ids is empty, set to None
        if not to_grow_ids:
            to_grow_ids = None
       
        if not self.early_stop_checkbox.isChecked():
            no_growth_max_iter = None
        else:
            no_growth_max_iter = self.no_growth_spin.value()
        
        return {
            "save_every_n_iters": self.save_every_n_spin.value() if self.save_iter_checkbox.isChecked() else None,
            "grow_to_end": self.grow_to_end_checkbox.isChecked(),
            "is_sort": self.sort_checkbox.isChecked(),
            "to_grow_ids": to_grow_ids,
            "min_growth_size": self.min_growth_spin.value(),
            "no_growth_max_iter": no_growth_max_iter
        }