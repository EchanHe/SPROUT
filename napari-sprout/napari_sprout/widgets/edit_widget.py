import sys
import weakref
import numpy as np
from skimage.measure import label as cc_label
from qtpy.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel,
                            QHBoxLayout, QCheckBox, QMessageBox,QSpinBox, QGroupBox,
                            QFormLayout, QScrollArea , QSizePolicy)
from napari.layers import Labels
from napari.utils.notifications import show_info, show_error
from qtpy.QtWidgets import QFrame

import numpy as np
from skimage.morphology import remove_small_holes
from skimage.morphology import erosion, dilation, opening, closing, disk, ball
from scipy import ndimage as ndi

from sprout_core.sprout_core import ensure_label_capacity, get_sub_binary_image_by_pos

from ..utils.status import LAYER_HEADER, NO_LAYER, StatusLine, style
from ..utils.layer_lock import lock_layers, lock_note, placement, unlock_layers
from ..utils.layer_shapes import current_layer_name, find_layer_item, grey_out_mismatches, placement_differs
from ..utils.roi_layers import get_roi_hub
from . import click_owner

# One style for every run button, the same as the main panel's run buttons. Only the enabled
# state is coloured, so a disabled run button keeps napari's disabled look.
RUN_STYLE = "QPushButton { font-weight: bold; } QPushButton:enabled { background-color: #45a049; }"
NO_TARGET = "—"   # shown by a Target Region box until a region ID is entered

def sort_labels_by_size(label_img, ignore_label=0):
    # Single pass over the array with bincount instead of one full scan per label.
    max_label = int(label_img.max())
    counts = np.bincount(label_img.ravel(), minlength=max_label + 1)
    if 0 <= ignore_label <= max_label:
        counts[ignore_label] = 0

    # Labels actually present, ordered by descending size. Sorting on the negated
    # counts with a stable sort keeps equal-size labels in ascending-label order,
    # matching the previous (stable) implementation.
    present = np.nonzero(counts)[0]
    order = present[np.argsort(-counts[present], kind="stable")]

    remap = {int(old): new + 1 for new, old in enumerate(order)}

    # Relabel via a lookup table (single vectorized indexing pass).
    lut = np.zeros(max_label + 1, dtype=label_img.dtype)
    for old, new in remap.items():
        lut[old] = new
    new_img = lut[label_img]

    return new_img, remap


def label_boxes(label_img):
    """(label, bounding box) of every non-zero label, in ascending label order (as np.unique).

    One ndi.find_objects pass instead of one full-image scan per label. find_objects skips
    negative values and makes one entry per ID up to the maximum, so negative or very large IDs
    are first mapped to compact ones (np.unique, still one pass).
    """
    if label_img.size == 0:
        return []
    lo, hi = int(label_img.min()), int(label_img.max())
    if lo >= 0 and hi <= 2 ** 24:
        boxes = ndi.find_objects(label_img)
        return [(label_img.dtype.type(i + 1), box) for i, box in enumerate(boxes) if box is not None]
    values, inverse = np.unique(label_img, return_inverse=True)
    boxes = ndi.find_objects(inverse.reshape(label_img.shape) + 1)
    return [(v, box) for v, box in zip(values, boxes) if v != 0]


def grow_box(box, margin, shape):
    """The box widened by `margin` on every side, clipped at the image border."""
    return tuple(slice(max(b.start - margin, 0), min(b.stop + margin, n)) for b, n in zip(box, shape))


def outside_pieces(box, shape):
    """Sizes of the connected pieces of the image outside `box` (face connectivity)."""
    n = int(np.prod(shape))
    sides = [(i, lo, hi) for i, (b, size) in enumerate(zip(box, shape))
             for lo, hi in ((0, b.start), (b.stop, size)) if hi > lo]
    if len({i for i, _, _ in sides}) >= 2:      # slabs on two axes meet: one piece around the box
        return [n - int(np.prod([b.stop - b.start for b in box]))]
    # on one axis only, the box spans the image along the others: a slab on each side
    return [(hi - lo) * n // shape[i] for i, lo, hi in sides]


def fill_holes_in_box(label_img, result, lbl, box, area_threshold):
    """remove_small_holes for one label, computed in its box plus 1 voxel; writes into result.

    Identical to remove_small_holes on the whole image. The label lies inside its box, so the
    image outside the box is background and each piece of it lies in one component. A background
    component of the crop that does not reach the 1-voxel ring around the box is a whole component
    of the image (same size, same decision). One that reaches the ring holds a whole outside piece
    in the image; if every piece is larger than the threshold, none of these is filled, so fills
    the crop made there are dropped. Otherwise (a small pocket between the label and the image
    border) the whole image is used, as before.
    """
    crop = grow_box(box, 1, label_img.shape)
    mask = label_img[crop] == lbl
    new = remove_small_holes(mask, area_threshold=area_threshold) & ~mask
    ring = new.copy()
    ring[tuple(slice(b.start - c.start, b.stop - c.start) for b, c in zip(box, crop))] = False
    if ring.any():
        if min(outside_pieces(box, label_img.shape)) > area_threshold:
            parts, _ = ndi.label(new)            # the filled components (face connectivity)
            new &= ~np.isin(parts, np.unique(parts[ring]))
        else:
            crop = tuple(slice(0, n) for n in label_img.shape)
            mask = label_img == lbl
            new = remove_small_holes(mask, area_threshold=area_threshold) & ~mask
    result[crop][new] = lbl


def has_region(data, target):
    """True if `target` occurs in data. A slab-by-slab scan that stops at the first hit:
    no np.unique (a full sort) and no full-size temporary on the UI thread."""
    if data.ndim == 0 or data.shape[0] == 0:
        return False
    step = max(1, (1 << 22) // max(1, data[0].size))
    return any(np.any(data[i:i + step] == target) for i in range(0, data.shape[0], step))

class QtLabelSelector(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.selected_labels = set()
        self.original_data_backup = None
        self._backup_layer = None   # weakref to the layer original_data_backup belongs to
        self._busy = 0              # background operations running
        self.last_bound_layer = None

        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Label layer selection
        self.layer_label = QLabel(NO_LAYER)
        self.layer_label.setStyleSheet(LAYER_HEADER)
        self.layer_label.setWordWrap(True)
        self.layer_label.setToolTip("The Labels layer the operations below work on: the active layer in napari.")
        layout.addWidget(self.layer_label)

        # Undo button
        undo_btn = QPushButton("Undo")
        undo_btn.setToolTip("Undo the last operation, on the layer it changed (one step).")
        undo_btn.clicked.connect(self.undo_last_operation)
        self.undo_btn = undo_btn   # disabled while an operation runs (it would replace the layer's data)
        layout.addWidget(undo_btn)
        self.status = StatusLine(hide_empty=True)
        layout.addWidget(self.status)

        # Duplicate option
        # self.duplicate_combo = QComboBox()
        # self.duplicate_combo.addItems(["No", "Yes"])
        # self.duplicate_combo.setToolTip("Select 'Yes' to apply processing on a duplicate of the original layer.")
        # layout.addWidget(QLabel("Duplicate layer:"))
        # layout.addWidget(self.duplicate_combo)

        duplicate_layout = QHBoxLayout()

        duplicate_label = QLabel("Duplicate layer:")
        duplicate_label.setToolTip("If checked, processing will apply on a duplicated copy of the current label layer.")

        self.duplicate_checkbox = QCheckBox()
        self.duplicate_checkbox.setChecked(False)  # default unchecked
        self.duplicate_checkbox.setToolTip("Check to apply processing on a duplicated layer instead of the original.")

        duplicate_layout.addWidget(duplicate_label)
        duplicate_layout.addWidget(self.duplicate_checkbox)
        duplicate_layout.addStretch()  

        layout.addLayout(duplicate_layout)
        
        # self.layer_combo = QComboBox()
        # self.layer_combo.currentTextChanged.connect(self.on_layer_combo_changed)
        
        

                
        # layout.addWidget(self.layer_combo)
        
        # self.update_layer_list()

        # self.viewer.layers.events.inserted.connect(self.update_layer_list)
        # self.viewer.layers.events.removed.connect(self.update_layer_list)


        label_select_group = QGroupBox("Region Selection")
        label_select_group.setToolTip("Select regions with Ctrl+click in the viewer, then keep or remove them.")
        label_select_layout = QFormLayout()

        select_hint = QLabel("Ctrl+click a region in the viewer to select it; Ctrl+click it again to deselect. "
                             "With Label Mapping open too, the dock used last takes the click.")
        select_hint.setWordWrap(True)
        select_hint.setStyleSheet(style("info"))
        label_select_layout.addRow(select_hint)

        # Region display
        self.label_display = QLabel("Selected regions: []")
        self.label_display.setToolTip("The regions selected with Ctrl+click in the viewer.")
# layout.addWidget(self.label_display)
        label_select_layout.addRow(self.label_display)
        
        # Clear button
        clear_btn = QPushButton("Clear Selection")
        clear_btn.setToolTip("Clear all currently selected regions.")
        clear_btn.clicked.connect(self.clear_selection)

        label_select_layout.addRow(clear_btn)

        # Mode select
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Keep selected regions", "Remove selected regions"])
        self.mode_combo.setToolTip("Choose whether to keep or remove the selected regions during processing.")
        # layout.addWidget(QLabel("Mode:"))
        # layout.addWidget(self.mode_combo)
        
        label_select_layout.addRow(QLabel("Mode:") ,self.mode_combo)

        # Run button
        run_btn = QPushButton("Run processing")
        run_btn.setToolTip("Run the main region selection processing according to the chosen mode and options.")
        run_btn.setStyleSheet(RUN_STYLE)
        run_btn.clicked.connect(self.process_selected_operation)
        # layout.addWidget(run_btn)
        label_select_layout.addRow(run_btn)
        label_select_group.setLayout(label_select_layout)
        # layout.addWidget(label_select_group)

        # layout.addWidget(create_separator(horizontal=True))
        # Set class selection operation
        # layout.addWidget(QLabel("Replace Label"))
        
        # ---- Label Replace Group ----
        label_replc_group = QGroupBox("Region Replace")
        label_replc_group.setToolTip("Replace all voxels of a source region with a target region.")
        label_replc_layout = QFormLayout()


        self.src_spin = QSpinBox()
        self.src_spin.setMinimum(0)
        self.src_spin.setMaximum(10000)
        self.src_spin.setToolTip("Enter the source region (ID) you want to replace.")

        self.dst_spin = QSpinBox()
        self.dst_spin.setMinimum(0)
        self.dst_spin.setMaximum(10000)
        self.dst_spin.setToolTip("Enter the target region (ID) that will replace the source.")

        label_replc_layout.addRow(QLabel("Source Region:"), self.src_spin)
        label_replc_layout.addRow(QLabel("Target Region:"), self.dst_spin)
        
        # layout.addWidget(QLabel("Source Class:"))
        # layout.addWidget(self.src_spin)

        # layout.addWidget(QLabel("Target Class:"))
        # layout.addWidget(self.dst_spin)

        set_class_btn = QPushButton("Replace")
        set_class_btn.setStyleSheet(RUN_STYLE)
        set_class_btn.setToolTip("Replace all voxels of the source region with the target region.")
        set_class_btn.clicked.connect(self.set_class_operation)
        # layout.addWidget(set_class_btn)
        label_replc_layout.addRow(set_class_btn)
        label_replc_group.setLayout(label_replc_layout)
        # layout.addWidget(label_replc_group)

        # ---- Merge Labels Group ----
        label_merge_group = QGroupBox("Merge Label Layers")
        label_merge_layout = QFormLayout()

        self.merge_into_label = QLabel("")
        self.merge_into_label.setToolTip("The selected label layer. It is changed (or copied first, with Duplicate).")
        label_merge_layout.addRow(QLabel("Into:"), self.merge_into_label)

        self.merge_source_combo = QComboBox()
        self.merge_source_combo.setToolTip(
            "Layer to add. Same shape as the Into layer. Where both have a region, the Into layer stays.")
        self.merge_source_combo.currentIndexChanged.connect(self._warn_merge_placement)
        label_merge_layout.addRow(QLabel("Add:"), self.merge_source_combo)

        self.merge_ids_combo = QComboBox()
        self.merge_ids_combo.addItems(["Keep", "Shift up"])
        self.merge_ids_combo.setToolTip(
            "Keep: same ID = same region (two class maps).\n"
            "Shift up: new IDs above the Into layer's (two seed layers).")
        label_merge_layout.addRow(QLabel("Added IDs:"), self.merge_ids_combo)

        merge_btn = QPushButton("Run Merge")
        merge_btn.setStyleSheet(RUN_STYLE)
        merge_btn.setToolTip("Add the layer to the Into layer.")
        merge_btn.clicked.connect(self.merge_labels_operation)
        label_merge_layout.addRow(merge_btn)

        label_merge_group.setLayout(label_merge_layout)
        # layout.addWidget(label_merge_group)
        
        self.viewer.layers.events.inserted.connect(self._refresh_merge_source_combo)
        self.viewer.layers.events.removed.connect(self._refresh_merge_source_combo)
        get_roi_hub(self.viewer).layer_renamed.connect(self._on_layer_renamed)
        

        # ---- Label Split Group ----
        label_split_group = QGroupBox("Region Split")

        label_split_group.setToolTip("Split a selected region into connected components.")
        label_split_layout = QFormLayout()

        self.split_all_checkbox = QCheckBox("Split all non-background regions")
        self.split_all_checkbox.setChecked(False)
        self.split_all_checkbox.setToolTip("If checked, split all regions > 0 instead of only the target region.")
        label_split_layout.addRow(self.split_all_checkbox)

        self.split_class_spin = self._target_spin(10000, "The region ID to split into components.")
        # Grey out the single-region target while "split all" is active.
        self.split_all_checkbox.toggled.connect(self.split_class_spin.setDisabled)

        label_split_layout.addRow(QLabel("Target Region:"), self.split_class_spin)
        
        self.sort_checkbox = QCheckBox("Sort by size (reassign IDs by size)")
        self.sort_checkbox.setChecked(True)
        self.sort_checkbox.setToolTip("If checked, reassign IDs based on component size.")
        label_split_layout.addRow(self.sort_checkbox)
        # layout.addWidget(self.sort_checkbox)


        self.top_n_spin = QSpinBox()
        self.top_n_spin.setMinimum(0)
        self.top_n_spin.setMaximum(10000)
        self.top_n_spin.setToolTip("If set > 0, keep only the top N largest components.")
        # layout.addWidget(QLabel("Keep Top-N Components (0 = all):"))
        # layout.addWidget(self.top_n_spin)
        label_split_layout.addRow(QLabel("Keep Top-N Components (0 = all):"), self.top_n_spin)

        split_btn = QPushButton("Run Split Region")
        split_btn.setStyleSheet(RUN_STYLE)
        split_btn.setToolTip("Split selected region into connected components.")
        split_btn.clicked.connect(self.split_class_operation)
        label_split_layout.addRow(split_btn)
        label_split_group.setLayout(label_split_layout)
        # layout.addWidget(label_split_group)

        # ---- Keep Top-N Labels Group ----
        keep_label_group = QGroupBox("Keep Top-N Regions")
        keep_label_group.setToolTip("Keep the N largest regions based on total area/volume.")

        keep_layout = QFormLayout()

        self.keep_label_topn_spin = QSpinBox()
        self.keep_label_topn_spin.setMinimum(1)
        self.keep_label_topn_spin.setMaximum(10_000)
        self.keep_label_topn_spin.setValue(5)
        self.keep_label_topn_spin.setToolTip("Number of regions to keep (by total pixel count).")
        keep_layout.addRow(QLabel("Top-N Regions:"), self.keep_label_topn_spin)

        keep_label_btn = QPushButton("Run Keep Top-N Regions")
        keep_label_btn.setToolTip("Keep the N largest regions; all others become background (0).")
        keep_label_btn.setStyleSheet(RUN_STYLE)
        keep_label_btn.clicked.connect(self.run_keep_top_n_labels)
        keep_layout.addRow(keep_label_btn)

        keep_label_group.setLayout(keep_layout)
        # layout.addWidget(keep_label_group)


        # ---- Filter operations Group ----
        filter_group = QGroupBox("Filter Regions")
        filter_group.setToolTip("Remove small connected components or keep only the largest regions.")

        filter_layout = QFormLayout()

        self.filter_all_checkbox = QCheckBox("Apply to all non-background regions")
        self.filter_all_checkbox.setChecked(False)
        self.filter_all_checkbox.setToolTip("If checked, filter all regions > 0 instead of a single target region.")
        filter_layout.addRow(self.filter_all_checkbox)

        self.filter_target_label_spin = self._target_spin(10_000, "The region to filter.")
        self.filter_all_checkbox.toggled.connect(self.filter_target_label_spin.setDisabled)
        filter_layout.addRow(QLabel("Target Region:"), self.filter_target_label_spin)

        self.filter_min_size_spin = QSpinBox()
        self.filter_min_size_spin.setMinimum(0)
        self.filter_min_size_spin.setMaximum(1_000_000_000)
        self.filter_min_size_spin.setValue(0)
        self.filter_min_size_spin.setToolTip("Remove all components smaller than this size (0 = skip).")
        filter_layout.addRow(QLabel("Minimum Size:"), self.filter_min_size_spin)

        self.filter_top_n_spin = QSpinBox()
        self.filter_top_n_spin.setMinimum(0)
        self.filter_top_n_spin.setMaximum(10_000)
        self.filter_top_n_spin.setValue(0)
        self.filter_top_n_spin.setToolTip("Keep only the top-N largest components (0 = all).")
        filter_layout.addRow(QLabel("Keep Top-N:"), self.filter_top_n_spin)

        filter_btn = QPushButton("Run Filtering")
        filter_btn.setToolTip("Remove small components / keep the largest components of the target region(s).")
        filter_btn.setStyleSheet(RUN_STYLE)
        filter_btn.clicked.connect(self.run_filtering_operation)
        filter_layout.addRow(filter_btn)

        filter_group.setLayout(filter_layout)
        # layout.addWidget(filter_group)


        # ---- Fill holes operation ----
        fill_group = QGroupBox("Fill Holes")
        fill_group.setToolTip("Fill small holes inside regions.")

        fill_layout = QFormLayout()

        self.fill_all_checkbox = QCheckBox("Apply to all non-background regions")
        self.fill_all_checkbox.setChecked(False)
        self.fill_all_checkbox.setToolTip("If checked, fill holes in all regions > 0 instead of a single target region.")
        fill_layout.addRow(self.fill_all_checkbox)

        self.fill_target_spin = self._target_spin(10_000, "The region to process.")
        self.fill_all_checkbox.toggled.connect(self.fill_target_spin.setDisabled)
        fill_layout.addRow(QLabel("Target Region:"), self.fill_target_spin)

        self.fill_area_spin = QSpinBox()
        self.fill_area_spin.setMinimum(1)
        self.fill_area_spin.setMaximum(1_000_000_000)
        self.fill_area_spin.setValue(64)
        self.fill_area_spin.setToolTip("Maximum hole area to fill (in pixels/voxels).")
        fill_layout.addRow(QLabel("Area Threshold:"), self.fill_area_spin)

        self.fill_in_2d_checkbox = QCheckBox("Apply per-slice (2D)")
        self.fill_in_2d_checkbox.setChecked(False)
        self.fill_in_2d_checkbox.setToolTip("Apply hole filling on each 2D slice (if 3D).")
        fill_layout.addRow(self.fill_in_2d_checkbox)

        fill_btn = QPushButton("Run Fill Holes")
        fill_btn.setToolTip("Fill holes up to the area threshold inside the target region(s).")
        fill_btn.setStyleSheet(RUN_STYLE)
        fill_btn.clicked.connect(self.run_fill_holes_operation)
        fill_layout.addRow(fill_btn)

        fill_group.setLayout(fill_layout)
        # layout.addWidget(fill_group)

        # ---- Morphology Group ----
        

        morph_group = QGroupBox("Morphology Transform")
        morph_group.setToolTip("Apply morphological operations to regions.")

        morph_layout = QFormLayout()

        self.morph_all_checkbox = QCheckBox("Apply to all non-background regions")
        self.morph_all_checkbox.setChecked(False)
        self.morph_all_checkbox.setToolTip("If checked, apply the operation to all regions > 0 instead of a single target region.")
        morph_layout.addRow(self.morph_all_checkbox)

        self.morph_target_spin = self._target_spin(10_000, "The region to process.")
        self.morph_all_checkbox.toggled.connect(self.morph_target_spin.setDisabled)
        morph_layout.addRow(QLabel("Target Region:"), self.morph_target_spin)

        self.morph_op_combo = QComboBox()
        self.morph_op_combo.addItems(["Erode", "Dilate", "Open", "Close"])
        self.morph_op_combo.setToolTip("Choose morphological operation to apply.")
        morph_layout.addRow(QLabel("Operation:"), self.morph_op_combo)

        self.morph_kernel_spin = QSpinBox()
        self.morph_kernel_spin.setMinimum(1)
        self.morph_kernel_spin.setMaximum(50)
        self.morph_kernel_spin.setValue(2)
        self.morph_kernel_spin.setToolTip("Kernel radius for the operation.")
        morph_layout.addRow(QLabel("Kernel Radius:"), self.morph_kernel_spin)

        morph_btn = QPushButton("Run Morphology")
        morph_btn.setToolTip("Apply the operation to the target region(s).")
        morph_btn.setStyleSheet(RUN_STYLE)
        morph_btn.clicked.connect(self.run_morphology_operation)
        morph_layout.addRow(morph_btn)

        morph_group.setLayout(morph_layout)
        # layout.addWidget(morph_group)


        content_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        content_widget.setMinimumWidth(content_widget.sizeHint().width())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)

        # 3. 最外层 layout 包裹 scroll
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(scroll)
        self.setLayout(outer_layout)

        # self.setLayout(layout)

        # Track layer selection
        self.viewer.layers.selection.events.active.connect(self.update_active_label_layer_binding)
        

        self.all_edit_buttons = [
            fill_btn, morph_btn, split_btn, filter_btn,
            keep_label_btn, run_btn, set_class_btn,
            merge_btn
        ]

        # A run button whose Target Region box is still empty is disabled ("all" ticked: enabled).
        self._target_buttons = [
            (split_btn, self.split_all_checkbox, self.split_class_spin),
            (filter_btn, self.filter_all_checkbox, self.filter_target_label_spin),
            (fill_btn, self.fill_all_checkbox, self.fill_target_spin),
            (morph_btn, self.morph_all_checkbox, self.morph_target_spin),
        ]
        for _, all_box, spin in self._target_buttons:
            all_box.toggled.connect(self._sync_target_buttons)
            spin.valueChanged.connect(self._sync_target_buttons)
        self._sync_target_buttons()


        tab_widget = QTabWidget()

        # ── Tab 1: Select & Edit ──────────────────────────
        # Region identity / membership: select, replace IDs, merge layers, split.
        tab_edit = QWidget()
        tab_edit_layout = QVBoxLayout(tab_edit)
        tab_edit_layout.addWidget(label_select_group)
        tab_edit_layout.addWidget(label_replc_group)
        tab_edit_layout.addWidget(label_merge_group)
        tab_edit_layout.addWidget(label_split_group)
        tab_edit_layout.addStretch()
        tab_widget.addTab(tab_edit, "Select && Edit")

        # ── Tab 2: Cleanup & Refine ───────────────────────
        # Noise removal and shape refinement: keep top-N, filter, fill, morphology.
        tab_refine = QWidget()
        tab_refine_layout = QVBoxLayout(tab_refine)
        tab_refine_layout.addWidget(keep_label_group)
        tab_refine_layout.addWidget(filter_group)
        tab_refine_layout.addWidget(fill_group)
        tab_refine_layout.addWidget(morph_group)
        tab_refine_layout.addStretch()
        tab_widget.addTab(tab_refine, "Cleanup && Refine")

        layout.addWidget(tab_widget)

        # Bind the layer that is already active (no re-selection needed).
        self._bind_layer(self.viewer.layers.selection.active)
        click_owner.register(self)   # with Label Mapping open too, the dock used last takes Ctrl+click

    def _target_spin(self, maximum, tip):
        """A Target Region box: empty (NO_TARGET) until a region ID is entered."""
        spin = QSpinBox()
        spin.setMinimum(-1)
        spin.setMaximum(maximum)
        spin.setSpecialValueText(NO_TARGET)
        spin.setValue(-1)
        spin.setToolTip(tip + " Empty until you enter a region ID; 0 is the background.")
        return spin

    def _sync_target_buttons(self, *args):
        """Enable each run button that has a target (a region ID entered, or "all" ticked)."""
        if self._busy:
            return
        for btn, all_box, spin in self._target_buttons:
            btn.setEnabled(all_box.isChecked() or spin.value() >= 0)

    def _no_target(self, target):
        """True (and a red line) if no Target Region was entered."""
        if target is not None and target < 0:
            self.status.set("Enter a target region first.", "error")
            return True
        return False

    def _set_backup(self, layer, data):
        """Remember `data` as the state Undo restores, for `layer` only."""
        self.original_data_backup = data
        self._backup_layer = weakref.ref(layer)

    def update_layer_list(self, *args):
        # deprecated method
        
        # update the combo box with current label layers
        self.layer_combo.blockSignals(True) 
        
        self.layer_combo.clear()
        label_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]
        self.layer_combo.addItems(label_layers)

        self.layer_combo.blockSignals(False)


    def on_layer_combo_changed(self, layer_name):
        # If the selected layer is a Labels layer, update the class combos
        layer = self.viewer.layers[layer_name]
        if isinstance(layer, Labels):
            
            # Bind the click handler to the selected layer
            if self.last_bound_layer and self.on_click in self.last_bound_layer.mouse_drag_callbacks:
                self.last_bound_layer.mouse_drag_callbacks.remove(self.on_click)
            if self.on_click not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = layer
        else:
            if self.last_bound_layer and self.on_click in self.last_bound_layer.mouse_drag_callbacks:
                self.last_bound_layer.mouse_drag_callbacks.remove(self.on_click)
            self.last_bound_layer = None
            
   

    def get_selected_layer(self) -> Labels:
        # selected_name = self.layer_combo.currentText()
        # return self.viewer.layers[selected_name]

        layer_name = self.combo.currentText()
        layer = self.viewer.layers.get(layer_name, None)
        if isinstance(layer, Labels):
            return layer
        return None

    def update_display_label(self):
        label_str = ", ".join(str(lbl) for lbl in sorted(self.selected_labels))
        self.label_display.setText(f"Selected regions: {label_str}")

    def on_click(self, layer, event):
        if not isinstance(layer, Labels):
            return
        if "Control" not in event.modifiers:
            return
        if not click_owner.handles_click(self):
            return

        label = layer.get_value(event.position,
                                view_direction=event.view_direction,
                                dims_displayed=event.dims_displayed)
        if label is None or label == 0:
            return

        if label in self.selected_labels:
            self.selected_labels.remove(label)
        else:
            self.selected_labels.add(label)

        self.update_display_label()


    def set_class_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        src = self.src_spin.value()
        dst = self.dst_spin.value()
        
        # if src == dst:return
        if src == dst:
            QMessageBox.information(self, "Info", "Source and target region are the same. No operation needed.")
            return
        
        # check if src value exists before doing calculations
        if not self.check_target_label(src, label_layer.data):
            return
        
        # Decide whether to duplicate the layer
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            duplicated_data = label_layer.data.copy()
            new_name = f"{label_layer.name}_set"
            target_layer = self.viewer.add_labels(duplicated_data, name=new_name, **placement(label_layer))
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
        # The edit goes into a new array, so the old one is the Undo state as it is (no copy).
        data = target_layer.data.copy()
        self._set_backup(target_layer, target_layer.data)
        data[data == src] = dst
        target_layer.data = data
        # QMessageBox.information(self, "Success", f"Class {src} changed to {dst}.")
        show_info(f"Region {src} changed to {dst} in layer {target_layer.name}.")

        self.clear_selection()

    ### commented out code for split_class_operation,   see below for the new implementation
    # def split_class_operation(self):
    #     label_layer = self.last_bound_layer
    #     if label_layer is None:
    #         QMessageBox.warning(self, "Error", "No label layer selected.")
    #         return

    #     target_class = self.split_class_spin.value()
    #     keep_top_n = self.top_n_spin.value()
    #     sort_by_size = self.sort_checkbox.isChecked()

    #     data = label_layer.data.copy()
    #     mask = (data == target_class)
    #     if not np.any(mask):
    #         QMessageBox.information(self, "Info", f"Class {target_class} not found.")
    #         return

    #     # Connected component labeling
    #     cc = cc_label(mask, connectivity=2)  
        
    #     num_components = cc.max()

    #     if num_components <= 1:
    #         QMessageBox.information(self, "Info", "No split needed.")
    #         return


    #     # Decide whether to duplicate the layer
    #     target_layer = label_layer
    #     if self.duplicate_checkbox.isChecked():
    #         duplicated_data = label_layer.data.copy()
    #         new_name = f"{label_layer.name}_split"
    #         target_layer = self.viewer.add_labels(duplicated_data, name=new_name)
    #         print(f"Operating on duplicated layer: {new_name}")
    #         if self.on_click not in target_layer.mouse_drag_callbacks:
    #             target_layer.mouse_drag_callbacks.append(self.on_click)
    #         self.last_bound_layer = target_layer

    #     # Optional: keep only top-N components
    #     if keep_top_n > 0:
    #         sizes = [(cc == i).sum() for i in range(1, num_components + 1)]
    #         sorted_indices = np.argsort(sizes)[::-1]
    #         keep_ids = set(sorted_indices[:keep_top_n] + 1)
    #         cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)

    #     # update the num_components after filtering
    #     num_components = cc.max()

    #     data[mask] = 0 
    #     new_label_base = data.max()
    #     for i in range(1, num_components + 1):
    #         data[cc == i] = i + new_label_base
            
    #     if sort_by_size:
    #         data,_ = sort_labels_by_size(data)


    #     # Update layer
    #     self.original_data_backup = target_layer.data.copy()
    #     target_layer.data = data
    #     QMessageBox.information(self, "Success", f"Split class {target_class} into {cc.max()} components.")
        
    #     self.clear_selection()


    def split_class_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        target_class = self.split_class_spin.value()
        keep_top_n = self.top_n_spin.value()
        sort_by_size = self.sort_checkbox.isChecked()
        if not self.split_all_checkbox.isChecked() and self._no_target(target_class):
            return

        # Warn when splitting region 0 in single-region mode: 0 is usually the
        # background, so splitting it tends to be slow and produce odd results.
        if not self.split_all_checkbox.isChecked() and target_class == 0:
            if not self._confirm_region_zero("Splitting"):
                return

        # check if target label exists before doing calculations (not for "split all": the
        # greyed-out Target Region may be empty or stale)
        split_all = self.split_all_checkbox.isChecked()
        if not self.check_target_label(None if split_all else target_class, label_layer.data):
            return

        # With "Duplicate layer" the "_split" copy is made only once the worker found
        # something to split (no leftover copy for "No split needed").
        # The worker builds a new array, so the layer's own array is the Undo state as it is;
        # with "Duplicate layer" it stays in the source layer, so the duplicate's Undo state is a copy.
        duplicate = self.duplicate_checkbox.isChecked()
        data = label_layer.data.copy() if duplicate else label_layer.data
        if not duplicate:
            self._set_backup(label_layer, data)

        def on_done(result):
            new_data, did_split = result
            if label_layer not in self.viewer.layers:
                self.status.set("The layer was removed before the operation finished.", "warn")
                return
            # The "no split needed" check now happens inside the worker, so the
            # connected-component labeling runs only once (off the UI thread).
            if not did_split:
                QMessageBox.information(self, "Info", "No split needed.")
                return
            target_layer = label_layer
            if duplicate:
                target_layer = self.viewer.add_labels(new_data, name=f"{label_layer.name}_split", **placement(label_layer))
                if self.on_click not in target_layer.mouse_drag_callbacks:
                    target_layer.mouse_drag_callbacks.append(self.on_click)
                self.last_bound_layer = target_layer
                self._set_backup(target_layer, data)
            else:
                target_layer.data = new_data
            if self.split_all_checkbox.isChecked():
                msg = f"Split all non-background regions completed in layer {target_layer.name}."
            else:
                msg = f"Split region {target_class} completed in layer {target_layer.name}."
            show_info(msg)
            self.clear_selection()

        if self.split_all_checkbox.isChecked():
            worker_func = self._do_split_all_classes
            worker_args = [data, keep_top_n, sort_by_size]
        else:
            worker_func = self._do_split_class
            worker_args = [data, target_class, keep_top_n, sort_by_size]

        self.run_in_background(
            worker_func,
            worker_args,
            on_done,
            buttons_to_disable=self.all_edit_buttons
        )

        # self.run_in_background(
        #     self._do_split_class,
        #     [data, target_class, keep_top_n, sort_by_size],
        #     on_done,
        #     buttons_to_disable= self.all_edit_buttons
        # )


    def _label_connected_components(self, mask: np.ndarray) -> np.ndarray:
        """Label connected components of a boolean mask.

        Crops to the mask's bounding box before labeling (mirrors sprout_core's
        get_sub_binary_image_by_pos usage) since the target region is often a
        small fraction of a large volume. Uses full connectivity (mask.ndim) so
        both 2D and 3D masks are handled correctly.
        """
        subset, bbox = get_sub_binary_image_by_pos(mask, margin=0)
        labeled_subset = cc_label(subset, connectivity=subset.ndim)

        labeled_full = np.zeros(mask.shape, dtype=labeled_subset.dtype)
        if mask.ndim == 3:
            min_z, max_z, min_y, max_y, min_x, max_x = bbox
            labeled_full[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = labeled_subset
        else:
            min_y, max_y, min_x, max_x = bbox
            labeled_full[min_y:max_y+1, min_x:max_x+1] = labeled_subset
        return labeled_full

    def _split_single_class(
        self,
        data: np.ndarray,
        target_class: int,
        keep_top_n: int,
        sort_by_size: bool,
        start_label_base: int,
    ):
        """Process a single target_class on data and return (result, new_label_base, did_split).

        - data: original label image
        - start_label_base: starting offset for numbering new components (usually pass current result.max())
        - did_split: True only if the class was actually split into >1 component
        """
        result = data.copy()
        mask = (result == target_class)
        if not np.any(mask):
            return result, start_label_base, False

        cc = self._label_connected_components(mask)
        num_components = cc.max()
        if num_components <= 1:
            return result, start_label_base, False

        if keep_top_n > 0:
            # bincount gives all component sizes in one pass; index [1:] skips background.
            sizes = np.bincount(cc.ravel(), minlength=num_components + 1)[1:]
            sorted_indices = np.argsort(sizes)[::-1]
            keep_ids = set(sorted_indices[:keep_top_n] + 1)
            cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)
            num_components = cc.max()

        # Vectorized relabel: every nonzero component id becomes base+id in one pass.
        result[mask] = 0
        base = start_label_base
        nonzero = cc > 0
        result[nonzero] = cc[nonzero] + base

        if sort_by_size:
            result, _ = sort_labels_by_size(result)
            base = result.max()
        else:
            base = base + num_components

        return result, base, True

    def _do_split_class(self, data: np.ndarray, target_class: int, keep_top_n: int, sort_by_size: bool):
        result, _, did_split = self._split_single_class(
            data=data,
            target_class=target_class,
            keep_top_n=keep_top_n,
            sort_by_size=sort_by_size,
            start_label_base=data.max()
        )
        return result, did_split

    def _do_split_all_classes(self, data: np.ndarray, keep_top_n: int, sort_by_size: bool):
        """Split every label of `data` into its connected components.

        Labels are taken in ascending order; the components of a split label get new IDs above
        the current maximum (as for a single region), each label is labelled inside its own
        bounding box. "Sort by size" renumbers all regions once at the end (it used to renumber
        the map after every split, so later labels of the original list hit other regions).
        """
        result = data.copy()
        boxes = label_boxes(data)

        if not boxes:
            return result, False

        # Starting base: current max label
        label_base = result.max()
        any_split = False

        for lbl, box in boxes:
            # IDs written so far are 0 or above the original maximum, so `data` still tells
            # which voxels are lbl.
            mask = data[box] == lbl
            cc = self._label_connected_components(mask)
            num_components = cc.max()
            if num_components <= 1:
                continue

            if keep_top_n > 0:
                sizes = np.bincount(cc.ravel(), minlength=num_components + 1)[1:]
                sorted_indices = np.argsort(sizes)[::-1]
                keep_ids = set(sorted_indices[:keep_top_n] + 1)
                cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)
                num_components = cc.max()

            if sort_by_size and label_base + num_components > np.iinfo(result.dtype).max:
                # Temporary IDs beyond the dtype: keep them exact until the final renumbering.
                result = result.astype(np.int64)
            view = result[box]
            view[mask] = 0
            nonzero = cc > 0
            view[nonzero] = cc[nonzero] + label_base
            label_base = label_base + num_components
            any_split = True

        if sort_by_size and any_split:
            result, _ = sort_labels_by_size(result)
            result = result.astype(data.dtype, copy=False)

        return result, any_split

    def update_active_label_layer_binding(self, event):
        self._bind_layer(event.value)

    def _bind_layer(self, layer):
        # unbind from the last layer if it exists
        if self.last_bound_layer and self.on_click in self.last_bound_layer.mouse_drag_callbacks:
            self.last_bound_layer.mouse_drag_callbacks.remove(self.on_click)
        self.last_bound_layer = None
        if isinstance(layer, Labels):
            if self.on_click not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = layer
            
            
            self.layer_label.setText(f"Active Label Layer: {layer.name}")
        else:
            self.layer_label.setText(NO_LAYER)
            
        self._refresh_merge_source_combo()

    def _on_layer_renamed(self, layer, old):
        """The header, Into and Add follow a renamed layer (a picked Add layer stays picked)."""
        if layer is self.last_bound_layer:
            self.layer_label.setText(f"Active Label Layer: {layer.name}")
        self._refresh_merge_source_combo(renamed=(old, layer.name))

    def _refresh_merge_source_combo(self, event=None, renamed=None):
        """Refresh merge source combo, excluding the current active layer."""
        current_selection = current_layer_name(self.merge_source_combo)
        if renamed is not None and current_selection == renamed[0]:
            current_selection = renamed[1]
        active_name = self.last_bound_layer.name if self.last_bound_layer else None
        self.merge_into_label.setText(active_name or "")

        self.merge_source_combo.blockSignals(True)
        self.merge_source_combo.clear()

        for layer in self.viewer.layers:
            if isinstance(layer, Labels) and layer.name != active_name:
                self.merge_source_combo.addItem(layer.name)

        # retain previous selection if possible
        idx = find_layer_item(self.merge_source_combo, current_selection)
        if idx >= 0:
            self.merge_source_combo.setCurrentIndex(idx)
        grey_out_mismatches(self.merge_source_combo, self.viewer, self.last_bound_layer)

        self.merge_source_combo.blockSignals(False)

    def _warn_merge_placement(self, *_):
        name, into = current_layer_name(self.merge_source_combo), self.last_bound_layer
        if into is not None and name in self.viewer.layers and placement_differs(self.viewer.layers[name], into):
            self.status.set("Add: scale/translate differs from the Into layer.", "warn")

    def clear_selection(self):
        self.selected_labels.clear()
        self.update_display_label()

    def undo_last_operation(self):
        layer = self._backup_layer() if self._backup_layer is not None else None
        if self.original_data_backup is None or self._backup_layer is None:
            self.status.set("Nothing to undo.", "info")
            return
        if layer is None or layer not in self.viewer.layers:
            # never write the backup into another layer
            self.original_data_backup = None
            self._backup_layer = None
            self.status.set("Nothing undone: the layer of the last operation was removed.", "warn")
            return
        # The backup is no longer used anywhere else (no operation writes into it): hand it back.
        layer.data = self.original_data_backup
        self.original_data_backup = None
        self._backup_layer = None
        other = layer is not self.viewer.layers.selection.active
        self.status.set(f"Undo: layer {layer.name} restored"
                        + (" (the layer of the last operation, not the selected one)." if other else "."), "ok")

    def process_selected_operation(self):
        if not self.selected_labels:
            self.status.set("No regions selected: Ctrl+click regions in the viewer first.", "warn")
            return

        label_layer = self.last_bound_layer

        # label_layer = self.get_selected_layer()
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No valid label layer selected.")
            return


        target_layer = label_layer
        # if self.duplicate_combo.currentText() == "Yes":
        if self.duplicate_checkbox.isChecked():
            duplicated_data = label_layer.data.copy()
            new_name = f"{label_layer.name}_filtered"
            target_layer = self.viewer.add_labels(duplicated_data, name=new_name, **placement(label_layer))
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer

        # data * mask below is a new array: the old one is the Undo state as it is.
        data = target_layer.data
        self._set_backup(target_layer, data)
        mask = np.isin(data, list(self.selected_labels))

        if self.mode_combo.currentText() == "Keep selected regions":
            data = data * mask
        elif self.mode_combo.currentText() == "Remove selected regions":
            data = data * (~mask)

        target_layer.data = data

        kept_or_removed = "kept" if self.mode_combo.currentText() == "Keep selected regions" else "removed"
        label_str = ", ".join(str(lbl) for lbl in sorted(self.selected_labels))
        
        # QMessageBox.information(
        #     self,
        #     "Done",
        #     f"Label selection processing completed.\n"
        #     f"{kept_or_removed.capitalize()} labels: [{label_str}]"
        # )
        
        show_info(
            f"Region selection processing completed in layer {target_layer.name}.\n"
            f"{kept_or_removed.capitalize()} regions: [{label_str}]"
        )
        
        self.clear_selection()

    def run_fill_holes_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        area_threshold = self.fill_area_spin.value()
        apply_in_2d = self.fill_in_2d_checkbox.isChecked()
        if self.fill_all_checkbox.isChecked():
            target_label = None  # process all regions > 0
        else:
            target_label = self.fill_target_spin.value()
            if self._no_target(target_label):
                return
            if target_label == 0 and not self._confirm_region_zero("Filling holes in"):
                return

        data = label_layer.data

        ## Check if target label exists before doing calculations
        if not self.check_target_label(target_label, data):
            return

        # Duplicate if needed
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_fill"
            target_layer = self.viewer.add_labels(data.copy(), name=new_name, **placement(label_layer))
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
            data = target_layer.data
        # The worker builds a new array: the current one is the Undo state as it is.
        self._set_backup(target_layer, data)  # for undo

        # Define a callback to run the long task in background
        def on_done(result):
            if target_layer not in self.viewer.layers:
                self.status.set("The layer was removed before the operation finished.", "warn")
                return
            target_layer.data = result
            # QMessageBox.information(self, "Done", "Hole filling completed.")
            show_info(f"Hole filling completed in layer {target_layer.name}.")

        # Run the long task in background
        self.run_in_background(
            self.fill_holes_in_labels,
            [data, area_threshold, target_label, apply_in_2d],
            on_done,
            buttons_to_disable=self.all_edit_buttons
        )


    def merge_labels_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No active label layer.")
            return

        source_name = current_layer_name(self.merge_source_combo)
        if not source_name:
            QMessageBox.warning(self, "Error", "Pick a layer to add.")
            return

        try:
            source_layer = self.viewer.layers[source_name]
        except KeyError:
            QMessageBox.warning(self, "Error", f"Layer '{source_name}' not found.")
            return

        if not isinstance(source_layer, Labels):
            QMessageBox.warning(self, "Error", f"'{source_name}' is not a Labels layer.")
            return

        target_data = label_layer.data
        source_data = source_layer.data

        if target_data.shape != source_data.shape:
            QMessageBox.warning(
                self, "Shape Mismatch",
                f"Into {target_data.shape}, Add {source_data.shape}."
            )
            return

        # Duplicate if needed
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_merged"
            new_layer = self.viewer.add_labels(target_data.copy(), name=new_name, **placement(label_layer))
            if self.on_click not in new_layer.mouse_drag_callbacks:
                new_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = new_layer
            label_layer = new_layer

        # The worker builds a new array: the current one is the Undo state as it is.
        data = label_layer.data
        self._set_backup(label_layer, data)

        use_offset = self.merge_ids_combo.currentText() == "Shift up"

        def on_done(result):
            if label_layer not in self.viewer.layers:
                self.status.set("The layer was removed before the operation finished.", "warn")
                return
            label_layer.data = result
            show_info(f"Merged '{source_name}' into '{label_layer.name}'.")

        self.run_in_background(
            self._do_merge_labels,
            [data, source_data, use_offset],
            on_done,
            buttons_to_disable=self.all_edit_buttons
        )

    def _do_merge_labels(
        self, target: np.ndarray, source: np.ndarray, use_offset: bool
    ) -> np.ndarray:
        """The source fills the 0 voxels of the target. use_offset: source ids start above the
        target's largest id. The result dtype holds every id (no uint8 wrap-around)."""
        fill = (target == 0) & (source != 0)
        offset = int(target.max()) if use_offset and target.size else 0
        added = source[fill].astype(np.int64) + offset
        max_id = max(int(target.max()) if target.size else 0, int(added.max()) if added.size else 0)
        result = ensure_label_capacity(target.copy(), max_id)
        result[fill] = added
        return result
    
    def fill_holes_in_labels(self, label_img, area_threshold=64, target_label=None, apply_in_2d=False):
        result = label_img.copy()
        ndim = result.ndim

        if target_label is not None:
            mask = (label_img == target_label)
            if mask.sum() == 0:
                return result
            
            max_area = np.prod(mask.shape) * 0.25
            current_area_threshold = min(area_threshold, max_area)

            if apply_in_2d and ndim == 3:
                for z in range(mask.shape[0]):
                    filled = remove_small_holes(mask[z], area_threshold=current_area_threshold)
                    result[z][filled & (~mask[z])] = target_label
            else:
                filled = remove_small_holes(mask, area_threshold=current_area_threshold)
                result[filled & (~mask)] = target_label
        else:
            # Each label inside its bounding box (fill_holes_in_box), in ascending order as before.
            max_area = np.prod(label_img.shape) * 0.25
            current_area_threshold = min(area_threshold, max_area)
            per_slice = apply_in_2d and ndim == 3
            if per_slice:
                # A slice without the label is one background component: filled whole when it
                # is smaller than the threshold (as remove_small_holes does on such a slice).
                fill_empty = remove_small_holes(np.zeros(label_img.shape[1:], bool),
                                                area_threshold=current_area_threshold).any()
            for lbl, box in label_boxes(label_img):
                if per_slice:
                    if fill_empty:
                        for z in range(label_img.shape[0]):
                            if not box[0].start <= z < box[0].stop:
                                result[z] = lbl
                    for z in range(box[0].start, box[0].stop):
                        fill_holes_in_box(label_img[z], result[z], lbl, box[1:], current_area_threshold)
                else:
                    fill_holes_in_box(label_img, result, lbl, box, current_area_threshold)
        return result

    ###commented out code for morphology operation, use the one below instead
    # def run_morphology_operation(self):
    #     label_layer = self.last_bound_layer
    #     if label_layer is None:
    #         QMessageBox.warning(self, "Error", "No label layer selected.")
    #         return

    #     op_name = self.morph_op_combo.currentText()
    #     radius = self.morph_kernel_spin.value()
    #     target_label = self.morph_target_spin.value()
    #     if target_label == 0:
    #         target_label = None  # process all labels

    #     op_map = {
    #         "Erode": erosion,
    #         "Dilate": dilation,
    #         "Open": opening,
    #         "Close": closing
    #     }
    #     morph_func = op_map.get(op_name)
    #     if morph_func is None:
    #         QMessageBox.warning(self, "Error", f"Unsupported operation: {op_name}")
    #         return

    #     data = label_layer.data.copy()
    #     ndim = data.ndim

    #     # Define structuring element
    #     if ndim == 2:
    #         selem = disk(radius)
    #     elif ndim == 3:
    #         selem = ball(radius)
    #     else:
    #         QMessageBox.warning(self, "Error", "Only 2D or 3D data is supported.")
    #         return

    #     # Duplicate if requested
    #     target_layer = label_layer
    #     if self.duplicate_checkbox.isChecked():
    #         new_name = f"{label_layer.name}_morph"
    #         target_layer = self.viewer.add_labels(data.copy(), name=new_name)
    #         print(f"Operating on duplicated layer: {new_name}")
    #         if self.on_click not in target_layer.mouse_drag_callbacks:
    #             target_layer.mouse_drag_callbacks.append(self.on_click)
    #         self.last_bound_layer = target_layer
    #         data = target_layer.data.copy()

    #     self.original_data_backup = data.copy()

    #     result = np.zeros_like(data)

    #     if target_label is not None:
    #         mask = (data == target_label)
    #         if not np.any(mask):
    #             QMessageBox.information(self, "Info", f"Label {target_label} not found.")
    #             return
    #         transformed = morph_func(mask, selem)
    #         result[transformed] = target_label
    #     else:
    #         unique_labels = np.unique(data)
    #         for lbl in unique_labels:
    #             if lbl == 0:
    #                 continue
    #             mask = (data == lbl)
    #             if not np.any(mask):
    #                 continue
    #             transformed = morph_func(mask, selem)
    #             result[transformed] = lbl

    #     target_layer.data = result
    #     QMessageBox.information(self, "Done", f"{op_name} operation completed.")

    def _confirm_region_zero(self, op_name="Processing"):
        """Ask before operating on region 0 (usually the background).

        Returns True if the user chose to proceed.
        """
        resp = QMessageBox.question(
            self,
            "Use region 0 (background)?",
            "You selected region 0, which is usually the background.\n"
            f"{op_name} it may be slow and produce strange results.\n\n"
            "Continue anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return resp == QMessageBox.Yes

    def check_target_label(self,target_label, data):
        # if target label is None, it means process all labels > 0, so no need to check existence
        # A scan that stops at the first hit (np.unique sorted the whole image on the UI thread);
        # the list of existing regions is computed only for the warning.
        if target_label is not None and not has_region(data, target_label):
            existing = np.unique(data)
            QMessageBox.warning(
                self,
                "Region Not Found",
                f"Region {target_label} does not exist in the current layer.\n"
                f"Existing regions: {sorted(existing[existing != 0].tolist())}"
            )
            return False
        return True

    def run_morphology_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        op_name = self.morph_op_combo.currentText()
        radius = self.morph_kernel_spin.value()
        if self.morph_all_checkbox.isChecked():
            target_label = None
        else:
            target_label = self.morph_target_spin.value()
            if self._no_target(target_label):
                return
            if target_label == 0 and not self._confirm_region_zero(f"Applying {op_name} to"):
                return

        data = label_layer.data
        ndim = data.ndim

        ## Check if target label exists before doing calculations
        if not self.check_target_label(target_label, data):
            return


        if ndim == 2:
            selem = disk(radius)
        elif ndim == 3:
            selem = ball(radius)
        else:
            QMessageBox.warning(self, "Error", "Only 2D or 3D data is supported.")
            return

        # The worker builds a new array: the current one is the Undo state as it is.
        self._set_backup(label_layer, data)

        # callback to run the long task in background
        def on_done(result):
            if self.duplicate_checkbox.isChecked():
                new_name = f"{label_layer.name}_morph"
                new_layer = self.viewer.add_labels(result, name=new_name, **placement(label_layer))
                if self.on_click not in new_layer.mouse_drag_callbacks:
                    new_layer.mouse_drag_callbacks.append(self.on_click)
                self.last_bound_layer = new_layer
                # Undo of a duplicated result works on the new layer (a copy: the
                # source array stays in use by the source layer)
                self._set_backup(new_layer, data.copy())
            else:
                if label_layer not in self.viewer.layers:
                    self.status.set("The layer was removed before the operation finished.", "warn")
                    return
                label_layer.data = result
                self.last_bound_layer = label_layer

            show_info(f"{op_name} operation completed.")

        # background task to perform morphology
        self.run_in_background(
            self._do_morphology,
            [data, target_label, op_name, selem],
            on_done,
            buttons_to_disable= self.all_edit_buttons
        )


    def _do_morphology(self, data: np.ndarray, target_label: int, op_name: str, selem) -> np.ndarray:
        op_map = {
            "Erode": erosion,
            "Dilate": dilation,
            "Open": opening,
            "Close": closing
        }
        morph_func = op_map.get(op_name)
        if morph_func is None:
            raise ValueError(f"Unsupported operation: {op_name}")

        result = np.zeros_like(data)

        
        
        if target_label is not None:
            mask = (data == target_label)
            if not np.any(mask):
                return result
            transformed = morph_func(mask, selem)
            result[transformed] = target_label

            preserve_mask = (data != 0) & (data != target_label)
            result[preserve_mask] = data[preserve_mask]

        else:
            # Each label inside its bounding box plus a margin: the result of a label lies within
            # r of it, and the crop edge (padded by reflection like the image border) only
            # reaches r further in, through background (2r for two-step Open / Close).
            r = max(selem.shape) // 2
            margin = (2 * r if op_name in ("Open", "Close") else r) + 1
            for lbl, box in label_boxes(data):
                crop = grow_box(box, margin, data.shape)
                transformed = morph_func(data[crop] == lbl, selem)
                result[crop][transformed] = lbl

        return result

    ### commented out code for filtering operation, use the one below instead
    # def run_filtering_operation(self):
    #     label_layer = self.last_bound_layer
    #     if label_layer is None:
    #         QMessageBox.warning(self, "Error", "No label layer selected.")
    #         return

    #     min_size = self.filter_min_size_spin.value()
    #     top_n = self.filter_top_n_spin.value()
    #     target_label = self.filter_target_label_spin.value()
    #     if target_label == 0:
    #         target_label = None

    #     if min_size == 0 and top_n == 0:
    #         QMessageBox.information(self, "Info", "No filtering performed (min size and top-N both zero).")
    #         return

    #     data = label_layer.data.copy()
    #     ndim = data.ndim
    #     target_layer = label_layer

    #     if self.duplicate_checkbox.isChecked():
    #         new_name = f"{label_layer.name}_filtered"
    #         target_layer = self.viewer.add_labels(data.copy(), name=new_name)
    #         print(f"Operating on duplicated layer: {new_name}")
    #         if self.on_click not in target_layer.mouse_drag_callbacks:
    #             target_layer.mouse_drag_callbacks.append(self.on_click)
    #         self.last_bound_layer = target_layer
    #         data = target_layer.data.copy()

    #     self.original_data_backup = data.copy()
    #     result = data.copy()

    #     if target_label is not None:
    #         mask = (data == target_label)
    #         result = self._filter_single_label(mask, result, target_label, min_size, top_n, ndim)
    #     else:
    #         unique_labels = np.unique(data)
    #         for lbl in unique_labels:
    #             if lbl == 0:
    #                 continue
    #             mask = (data == lbl)
    #             result = self._filter_single_label(mask, result, lbl, min_size, top_n, ndim)

    #     target_layer.data = result
    #     QMessageBox.information(self, "Done", "Filtering completed.")

    def run_filtering_operation(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        min_size = self.filter_min_size_spin.value()
        top_n = self.filter_top_n_spin.value()
        if self.filter_all_checkbox.isChecked():
            target_label = None
        else:
            target_label = self.filter_target_label_spin.value()
            if self._no_target(target_label):
                return
            if target_label == 0 and not self._confirm_region_zero("Filtering"):
                return

        if min_size == 0 and top_n == 0:
            QMessageBox.information(self, "Info", "No filtering performed (min size and top-N both zero).")
            return

        data = label_layer.data
        ndim = data.ndim

        # check if target label exists before doing calculations
        if not self.check_target_label(target_label, data):
            return

        # Duplicate if needed
        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_filtered"
            target_layer = self.viewer.add_labels(data.copy(), name=new_name, **placement(label_layer))
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
            data = target_layer.data
        # The worker builds a new array: the current one is the Undo state as it is.
        self._set_backup(target_layer, data)

        # callback to run the long task in background
        def on_done(result):
            if target_layer not in self.viewer.layers:
                self.status.set("The layer was removed before the operation finished.", "warn")
                return
            target_layer.data = result
            # QMessageBox.information(self, "Done", "Filtering completed.")
            show_info(f"Filtering completed in layer {target_layer.name}.")

        # start the background task
        self.run_in_background(
            self._do_filtering,
            [data, target_label, min_size, top_n, ndim],
            on_done,
            buttons_to_disable= self.all_edit_buttons
        )
    

    def _do_filtering(self, data: np.ndarray, target_label: int, min_size: int, top_n: int, ndim: int) -> np.ndarray:
        result = data.copy()

        if target_label is not None:
            mask = (data == target_label)
            result = self._filter_single_label(mask, result, target_label, min_size, top_n, ndim)
        else:
            # Each label inside its bounding box: its components and writes all lie there.
            for lbl, box in label_boxes(data):
                mask = (data[box] == lbl)
                self._filter_single_label(mask, result[box], lbl, min_size, top_n, ndim)

        return result


    def _filter_single_label(self, mask, result, label_val, min_size, top_n, ndim):
        labeled = self._label_connected_components(mask)

        # Single-pass sizes for every component (index 0 is background).
        counts = np.bincount(labeled.ravel())
        component_ids = np.nonzero(counts)[0]
        component_ids = component_ids[component_ids != 0]  # skip background 0
        sizes = {int(i): int(counts[i]) for i in component_ids}

        # Remove small components
        keep_ids = [i for i, size in sizes.items() if size >= min_size] if min_size > 0 else list(component_ids)

        # Keep only top-N if specified
        if top_n > 0 and len(keep_ids) > top_n:
            sorted_by_size = sorted(keep_ids, key=lambda x: sizes[x], reverse=True)
            keep_ids = sorted_by_size[:top_n]

        keep_mask = np.isin(labeled, keep_ids)
        result[mask] = 0  # remove all current pixels of this label
        result[keep_mask] = label_val
        return result

    def run_keep_top_n_labels(self):
        label_layer = self.last_bound_layer
        if label_layer is None:
            QMessageBox.warning(self, "Error", "No label layer selected.")
            return

        top_n = self.keep_label_topn_spin.value()
        data = label_layer.data

        target_layer = label_layer
        if self.duplicate_checkbox.isChecked():
            new_name = f"{label_layer.name}_keeplabels"
            target_layer = self.viewer.add_labels(data.copy(), name=new_name, **placement(label_layer))
            if self.on_click not in target_layer.mouse_drag_callbacks:
                target_layer.mouse_drag_callbacks.append(self.on_click)
            self.last_bound_layer = target_layer
            data = target_layer.data
        
        # The worker builds a new array: the current one is the Undo state as it is.
        self._set_backup(target_layer, data)

        def on_done(done):
            # The regions are counted in the worker (np.unique sorted the whole image on the UI thread).
            result, n_regions = done
            if top_n >= n_regions:
                QMessageBox.information(self, "Info", f"Only {n_regions} regions found. No filtering needed.")
                return
            if target_layer not in self.viewer.layers:
                self.status.set("The layer was removed before the operation finished.", "warn")
                return
            target_layer.data = result
            # QMessageBox.information(self, "Done", f"Kept top {top_n} labels.")
            show_info(f"Kept top {top_n} regions in layer {target_layer.name}.")

        self.run_in_background(
            self._do_keep_top_n_labels,
            [data, top_n],
            on_done,
            buttons_to_disable=[self.keep_label_btn] if hasattr(self, "keep_label_btn") else []
        )

    def _do_keep_top_n_labels(self, data: np.ndarray, top_n: int):
        """(result, number of regions); the result is `data` itself when nothing is removed."""
        labels = np.unique(data)
        labels = labels[labels != 0]

        if top_n >= len(labels):
            return data, len(labels)  # 不需要处理

        # Single pass over the array instead of one full scan per label.
        counts = np.bincount(data.ravel())
        sizes = {int(lbl): int(counts[lbl]) for lbl in labels}
        sorted_labels = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

        keep_labels = [lbl for lbl, _ in sorted_labels[:top_n]]

        result = np.where(np.isin(data, keep_labels), data, 0)
        return result, len(labels)



    def run_in_background(self, func, args, done_callback, buttons_to_disable=None):
        from napari.qt.threading import thread_worker

        @thread_worker
        def worker_wrapper(*args):
            return func(*args)

        worker = worker_wrapper(*args)

        if buttons_to_disable is None:
            buttons_to_disable = []

        # set buttons to disabled state during the background task
        self._busy += 1
        self.status.clear()
        for btn in buttons_to_disable:
            btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        # the layers whose own array the worker reads: no brush strokes into it meanwhile
        locked = lock_layers(self.viewer, [a for a in args if isinstance(a, np.ndarray)])
        note = lock_note(locked)
        if note:
            self.status.set(note, "info")

        def finish():
            self._busy -= 1
            for btn in buttons_to_disable:
                btn.setEnabled(True)
            self.undo_btn.setEnabled(not self._busy)
            self._sync_target_buttons()

        def release():
            unlock_layers(locked)   # after the result is written (setting data resets editable)
            if note and self.status.text() == note:
                self.status.clear()

        def on_done(result):
            # when the background task is done, re-enable buttons
            finish()
            try:
                done_callback(result)
            finally:
                release()

        def on_error(e):
            finish()
            release()
            QMessageBox.warning(self, "Error", f"Operation failed:\n{e}")

        worker.returned.connect(on_done)
        worker.errored.connect(on_error)
        worker.start()