import json
import os
from qgis.PyQt.QtWidgets import (QTableWidget, QComboBox, QPushButton, QCheckBox,
                                 QVBoxLayout, QWidget, QHeaderView, QFileDialog, QHBoxLayout)
from qgis.PyQt.QtCore import Qt
from qgis.core import (QgsProcessingParameterDefinition,
                       QgsProcessingAlgorithm,
                       QgsProject, QgsRasterLayer,
                       QgsProcessingParameterString)
from qgis.gui import QgsAbstractProcessingParameterWidgetWrapper, QgsFileWidget


# Custom Widget for selecting pairs of rasters
class RasterPairWidgetWrapper(QgsAbstractProcessingParameterWidgetWrapper):
    def __init__(self, param, parent, *args, **kwargs):
        super().__init__(param, parent=parent)
        self.widget = None
        self.table = None

    def widget(self):
        return self._widget

    # Create the widget GUI
    def createWidget(self):
        self._widget = QWidget()
        layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Training', 'Target', 'Validation Only'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        btn_add = QPushButton("Add Pair")
        btn_add.clicked.connect(self.add_pair)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self.remove_pair)
        
        layout.addWidget(self.table)
        layout.addWidget(btn_add)
        layout.addWidget(btn_remove)
        self._widget.setLayout(layout)
        return self._widget

    # Add a new pair of combo boxes to the table
    def add_pair(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.create_combo_widget(row, 0)
        self.create_combo_widget(row, 1)
        self.create_checkbox_widget(row)

    # Emit the value changed signal
    def emit_value_changed(self):
        pass

    # Remove the selected pair from the table
    def remove_pair(self):
        if self.table.rowCount() > 0:
            current_row = self.table.currentRow() if self.table.currentRow() != -1 else self.table.rowCount() - 1
            self.table.removeRow(current_row)


    # Creates a file widget that is used in the combo boxes
    def create_file_widget(self, row, col):
        file_widget = QgsFileWidget()
        file_widget.setFilter("Raster files (*.tif *.tiff *.geotiff *.img *.jp2 *.hdr *.asc *.grd);;All files (*.*)")
        file_widget.setDialogTitle("Select Raster File")
        file_widget.setStorageMode(QgsFileWidget.StorageMode.File)
        file_widget.setDefaultRoot(QgsProject.instance().homePath())
        return file_widget


    # Adds all the raster layers in the project to the combo boxes
    def create_combo_widget(self, row, col):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        combo = QComboBox()
        combo.addItem("")
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer):
                combo.addItem(lyr.name(), lyr.source())
        
        btn = self.create_file_widget()
        
        layout.addWidget(combo)
        layout.addWidget(btn)
        widget.setLayout(layout)
        widget.combo = combo
        combo.currentIndexChanged.connect(self.emit_value_changed)
        self.table.setCellWidget(row, col, widget)

    def create_checkbox_widget(self, row):
        checkbox = QCheckBox()
        checkbox.stateChanged.connect(self.emit_value_changed)
        self.table.setCellWidget(row, 2, checkbox)

    def select_file(self, row, col):
        file_path, _ = QFileDialog.getOpenFileName(
            self._widget,
            "Select Raster File",
            "",
            "Raster Files (*.tif *.tiff *.geotiff *.img *.jp2 *.hdr *.asc *.grd)"
        )
        if file_path:
            widget = self.table.cellWidget(row, col)
            combo = widget.combo
            idx = combo.findData(file_path)
            if idx == -1:
                combo.addItem(os.path.basename(file_path), file_path)
                idx = combo.count() - 1
            combo.setCurrentIndex(idx)



    # set the values of the rasters in the widget based on a json string
    def setWidgetValue(self, value, context):
        self.table.setRowCount(0)
        if value:
            try:
                pairs = json.loads(value)
                for pair in pairs:
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    self.create_combo_widget(row, 0)
                    self.create_combo_widget(row, 1)
                    self.create_checkbox_widget(row)

                    if len(pair) >= 3:
                        training_source, target_source, validation = pair[:3]
                    else:
                        training_source, target_source = pair[:2]
                        validation = False

                    self.set_combo_value(row, 0, training_source)
                    self.set_combo_value(row, 1, target_source)
                    self.table.cellWidget(row, 2).setChecked(validation)
            except json.JSONDecodeError:
                pass

    def set_combo_value(self, row, col, value):
        if not value:
            return
        widget = self.table.cellWidget(row, col)
        combo = widget.combo
        idx = combo.findData(value)
        if idx == -1:
            combo.addItem(os.path.basename(value), value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def widgetValue(self):
        pairs = []
        for row in range(self.table.rowCount()):
            training_widget = self.table.cellWidget(row, 0)
            target_widget = self.table.cellWidget(row, 1)
            training = training_widget.combo.currentData()
            target = target_widget.combo.currentData()
            validation = self.table.cellWidget(row, 2).isChecked()
            
            if training and target:
                pairs.append([training, target, validation])
        return json.dumps(pairs) if pairs else ""


class RasterPairParameter(QgsProcessingParameterDefinition):
    def __init__(self, name, description=""):
        super().__init__(name, description)
        self.setMetadata({
            'widget_wrapper': {
                'class': RasterPairWidgetWrapper
            }
        })

    def clone(self):
        return RasterPairParameter(self.name(), self.description())

    def type(self):
        return "raster_pair"

    def checkValueIsAcceptable(self, value, context=None):
        return True