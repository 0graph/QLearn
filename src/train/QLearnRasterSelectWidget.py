import json
import os
from qgis.PyQt.QtWidgets import (QTableWidget, QComboBox, QPushButton, QCheckBox,
                                 QVBoxLayout, QWidget, QHeaderView, QHBoxLayout)
from qgis.PyQt.QtCore import Qt, QObject
from qgis.core import (QgsProcessingParameterDefinition,
                       QgsProject, QgsRasterLayer)
from qgis.gui import (QgsAbstractProcessingParameterWidgetWrapper,
                      QgsFileWidget)

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
        self.table.setHorizontalHeaderLabels(['Training', 'Target', 'Eval Only'])
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

    def create_combo_widget(self, row, col):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        combo = QComboBox()
        combo.addItem("")
        self.populate_project_layers(combo)
        
        file_widget = self.create_file_widget()
        
        # Connect signals for two-way binding
        file_widget.fileChanged.connect(
            lambda path, c=combo: self.update_combo_from_file(c, path))
        combo.currentIndexChanged.connect(
            lambda: self.update_file_widget_from_combo(combo, file_widget))
        
        layout.addWidget(combo)
        layout.addWidget(file_widget)
        widget.setLayout(layout)
        widget.combo = combo
        widget.file_widget = file_widget
        self.table.setCellWidget(row, col, widget)

    def create_file_widget(self):
        file_widget = QgsFileWidget()
        file_widget.setFilter("Raster files (*.tif *.tiff *.geotiff *.img *.jp2 *.hdr *.asc *.grd);;All files (*.*)")
        file_widget.setStorageMode(QgsFileWidget.GetFile)
        file_widget.setDialogTitle("Select Raster File")
        file_widget.setDefaultRoot(QgsProject.instance().homePath())
        # remove line edit from file widget
        file_widget.lineEdit().setVisible(False)
        # set widget width to 30
        file_widget.setMaximumWidth(30)
        return file_widget

    def populate_project_layers(self, combo):
        combo.clear()
        combo.addItem("")
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer):
                combo.addItem(lyr.name(), lyr.source())

    def update_combo_from_file(self, combo, path):
        if path:
            idx = combo.findData(path)
            if idx == -1:
                combo.addItem(os.path.basename(path), path)
                idx = combo.count() - 1
            combo.setCurrentIndex(idx)
        self.emit_value_changed()

    def update_file_widget_from_combo(self, combo, file_widget):
        path = combo.currentData()
        if path:
            file_widget.setFilePath(path)
        self.emit_value_changed()

    def create_checkbox_widget(self, row):
        widget = QWidget()
        checkbox = QCheckBox()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(checkbox)
        checkbox.stateChanged.connect(self.emit_value_changed)
        widget.setLayout(layout)
        self.table.setCellWidget(row, 2, widget)

    def setWidgetValue(self, value, context):
        self.table.setRowCount(0)
        if not value:
            return
        
        try:
            pairs = json.loads(value)
            for pair in pairs:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.create_combo_widget(row, 0)
                self.create_combo_widget(row, 1)
                self.create_checkbox_widget(row)

                if len(pair) >= 3:
                    training, target, validation = pair[:3]
                else:
                    training, target = pair[:2]
                    validation = False

                self.set_combo_value(row, 0, training)
                self.set_combo_value(row, 1, target)
                # Fix: Access the checkbox within the widget container
                checkbox_widget = self.table.cellWidget(row, 2)
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(validation)
        except json.JSONDecodeError:
            pass

    def set_combo_value(self, row, col, value):
        cell_widget = self.table.cellWidget(row, col)
        if cell_widget and value:
            combo = cell_widget.combo
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
            
            training = training_widget.combo.currentData() if training_widget else None
            target = target_widget.combo.currentData() if target_widget else None
            widget = self.table.cellWidget(row, 2)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                validation = checkbox.isChecked() if checkbox else False
            else:
                validation = False
            
            if training and target:
                pairs.append([training, target, validation])
        return json.dumps(pairs) if pairs else ""

    def emit_value_changed(self):
        pass

    def remove_pair(self):
        if self.table.rowCount() > 0:
            current_row = self.table.currentRow() if self.table.currentRow() != -1 else self.table.rowCount() - 1
            self.table.removeRow(current_row)

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