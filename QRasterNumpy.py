# Fix for older versions where QgsRasterLayer.as_numpy and QgsRasterBlock.as_numpy does not exist
# Source: https://github.com/qgis/QGIS/blob/master/python/PyQt6/core/__init__.py.in#L544
import typing as _typing

from qgis.PyQt.QtCore import NULL
from qgis.PyQt.QtCore import Qt as _Qt
from qgis._core import *

try:
   import numpy as _numpy
   def _qgis_data_type_to_numeric_data_type(dataType: Qgis.DataType) -> _typing.Optional[_numpy.dtype]:
      qgis_to_numpy_dtype_dict = {
       Qgis.DataType.UnknownDataType: None,
       Qgis.DataType.Byte: _numpy.byte,
       Qgis.DataType.Int8: _numpy.int8,
       Qgis.DataType.UInt16: _numpy.uint16,
       Qgis.DataType.Int16: _numpy.int16,
       Qgis.DataType.UInt32: _numpy.uint32,
       Qgis.DataType.Int32: _numpy.int32,
       Qgis.DataType.Float32: _numpy.float32,
       Qgis.DataType.Float64: _numpy.float64,
       Qgis.DataType.CInt16: None,
       Qgis.DataType.CInt32: None,
       Qgis.DataType.CFloat32: _numpy.complex64,
       Qgis.DataType.CFloat64: _numpy.complex128,
       Qgis.DataType.ARGB32: None,
       Qgis.DataType.ARGB32_Premultiplied: None
       }
      return qgis_to_numpy_dtype_dict[dataType]

   def _raster_block_as_numpy(self, use_masking:bool = True) -> _typing.Union[_numpy.ndarray, _numpy.ma.MaskedArray]:
      raster_dtype = _qgis_data_type_to_numeric_data_type(self.dataType())
      if not raster_dtype:
         raise ValueError(f"The raster block data type '{str(self.dataType())}' is not compatible with NumPy arrays.")
      src_array = _numpy.frombuffer(self.data(), dtype=raster_dtype)
      src_array = src_array.reshape((self.height(), self.width()))
      if use_masking:
         if not self.hasNoDataValue():
      	    # Default to -100 as noDataValue if none is set
            # Note: changed this from the QGIS implementation as I'm not sure why we'd want to mask out 0s if there is no NODATA value set
            no_data_value = -100
         else:
            no_data_value = self.noDataValue()
         return _numpy.ma.masked_equal(src_array, no_data_value)
      else:
         return src_array

   QgsRasterBlock.as_numpy = _raster_block_as_numpy

   def _raster_layer_as_numpy(self, use_masking=True, bands: _typing.Optional[_typing.List[int]] = None) -> _typing.List[_typing.Union[_numpy.ndarray, _numpy.ma.MaskedArray]]:
      arrays = []
      band_range = bands if bands else range(self.bandCount())

      for band in band_range:
          block = self.dataProvider().block(band + 1, self.extent(), self.width(), self.height())
          src_array = block.as_numpy(use_masking=use_masking)
          arrays.append(src_array)

      if use_masking:
          return _numpy.ma.stack(arrays, axis=0)
      else:
          return _numpy.array(arrays)

   QgsRasterLayer.as_numpy = _raster_layer_as_numpy

   def _qgsgeometry_as_numpy(self) -> _typing.Union[_numpy.ndarray, _typing.List[_numpy.ndarray]]:
       wkb_type = self.wkbType()
       hasM = QgsWkbTypes.hasM(wkb_type)
       hasZ = QgsWkbTypes.hasZ(wkb_type)
       geometry_type = self.type()

       def get_xyzm_coordinates(pt):
                 if hasZ and hasM:
                     return _numpy.array([pt.x(), pt.y(), pt.z(), pt.m()])
                 elif hasZ:
                     return _numpy.array([pt.x(), pt.y(), pt.z()])
                 elif hasM:
                     return _numpy.array([pt.x(), pt.y(), pt.m()])
                 else:
                     return _numpy.array([pt.x(), pt.y()])

       def fill_structure_with_elements(lst: _typing.List, elements: _typing.List, idx: int=0):
           for i in range(len(lst)):
               if isinstance(lst[i], list):
                    idx = fill_structure_with_elements(lst[i], elements, idx)
               else:
                   lst[i] = _numpy.array(elements[idx])
                   idx += 1
           return idx

       if self.isMultipart():
            elements = [get_xyzm_coordinates(i) for i in self.vertices()]

            if geometry_type == QgsWkbTypes.PointGeometry:
                skeleton = self.asMultiPoint()
                fill_structure_with_elements(skeleton, elements)
                return skeleton

            elif geometry_type == QgsWkbTypes.LineGeometry:
                skeleton = self.asMultiPolyline()
                fill_structure_with_elements(skeleton, elements)
                return skeleton

            elif geometry_type == QgsWkbTypes.PolygonGeometry:
                skeleton = self.asMultiPolygon()
                fill_structure_with_elements(skeleton, elements)
                return skeleton
       else:
            if geometry_type == QgsWkbTypes.PointGeometry:
                return _numpy.array([get_xyzm_coordinates(i) for i in self.vertices()][0])
            elif geometry_type == QgsWkbTypes.LineGeometry:
                line = self.vertices()
                return _numpy.array([get_xyzm_coordinates(pt) for pt in line])
            elif geometry_type == QgsWkbTypes.PolygonGeometry:
                skeleton = self.asPolygon()
                elements = [get_xyzm_coordinates(i) for i in self.vertices()]
                fill_structure_with_elements(skeleton, elements)
                return _numpy.array(skeleton)


   QgsGeometry.as_numpy = _qgsgeometry_as_numpy

except ModuleNotFoundError:
   def _raster_block_as_numpy(self, use_masking:bool = True):
       raise QgsNotSupportedException('QgsRasterBlock.as_numpy is not available, numpy is not installed on the system')

   QgsRasterBlock.as_numpy = _raster_block_as_numpy

   def _raster_layer_as_numpy(self, use_masking:bool = True, bands: _typing.Optional[_typing.List[int]] = None):
       raise QgsNotSupportedException('QgsRasterLayer.as_numpy is not available, numpy is not installed on the system')

   QgsRasterLayer.as_numpy = _raster_layer_as_numpy

   def _geometry_as_numpy(self):
      raise QgsNotSupportedException('QgsGeometry.as_numpy is not available, numpy is not installed on the system')

   QgsGeometry.as_numpy = _geometry_as_numpy