"""
Geometry — representación geométrica del húmero y transporte standard → paciente.

Módulos:
  lorentz.py      O(3,1): álgebra de Minkowski y el boost que fija la esfera
  similarity.py   grupo de similaridades T(x) = (sR)x + b y el invariante R/d
  selftest.py     verificación (incluye el ground truth de la presentación)

Este paquete NO depende de SAM, torch ni de las máscaras: trabaja sobre los CSV que
ya produce humerus_boundary_analysis.py y sobre los .stl de excercises/.
"""
