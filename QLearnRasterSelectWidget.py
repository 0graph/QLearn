import json
from qgis.PyQt.QtWidgets import (QTableWidget, QComboBox, QPushButton, 
                                 QVBoxLayout, QWidget, QHeaderView)
from qgis.core import (QgsProcessingParameterDefinition,
                       QgsProcessingAlgorithm,
                       QgsProject, QgsRasterLayer,
                       QgsProcessingParameterString)
from qgis.gui import QgsAbstractProcessingParameterWidgetWrapper

# TODO: Allow the user to select a raster from disk
# TODO: Add a checkbox for selecting pairs to be used only as validation data

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
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Training', 'Target'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
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
        self.populate_combos(row)

    # Remove the selected pair from the table
    def remove_pair(self):
        if self.table.rowCount() > 0:
            current_row = self.table.currentRow() if self.table.currentRow() != -1 else self.table.rowCount() - 1
            self.table.removeRow(current_row)

    # Adds all the raster layers in the project to the combo boxes
    def populate_combos(self, row):
        rasters = [lyr for lyr in QgsProject.instance().mapLayers().values()
                   if isinstance(lyr, QgsRasterLayer)]
        
        for col in range(2):
            combo = QComboBox()
            combo.addItem("")  # Empty selection
            for lyr in rasters:
                combo.addItem(lyr.name(), lyr.source())
            combo.currentIndexChanged.connect(self.emit_value_changed)
            self.table.setCellWidget(row, col, combo)

    def emit_value_changed(self):
        pass

    # set the values of the rasters in the widget based on a json string
    def setWidgetValue(self, value, context):
        self.table.setRowCount(0)
        if value:
            try:
                pairs = json.loads(value)
                for pair in pairs:
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    self._populate_combos(row)
                    for col, layer_id in enumerate(pair):
                        combo = self.table.cellWidget(row, col)
                        idx = combo.findData(layer_id)
                        combo.setCurrentIndex(idx if idx != -1 else 0)
            except json.JSONDecodeError:
                pass

    # get the values of the rasters in the widget as a json string
    def widgetValue(self):
        pairs = []
        for row in range(self.table.rowCount()):
            pair = [
                self.table.cellWidget(row, 0).currentData(),
                self.table.cellWidget(row, 1).currentData()
            ]
            if all(pair):
                pairs.append(pair)
        return json.dumps(pairs)


# Custom parameter definition for selecting pairs of rasters
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