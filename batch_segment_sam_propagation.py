#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch runner for segment_sam_propagation over multiple DICOM datasets.

For cada dataset definido en DATASET_NAMES, ejecuta el pipeline completo de
segment_sam_propagation.main(), usando como:

  data_dir  = ROOT/Datasets/In/<dataset_name>
  output_dir = ROOT/Datasets/Out/<dataset_name>

De esta forma, la salida de cada dataset queda en una carpeta de resultados
con el mismo nombre que la carpeta de origen, pero bajo Datasets/Out.
"""

import os
import segment_sam_propagation as ssp


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_ROOT = os.path.join(ROOT_DIR, "Datasets", "In")
OUT_ROOT = os.path.join(ROOT_DIR, "Datasets", "Out")

# Nombres EXACTOS de las carpetas de entrada dentro de Datasets/In
DATASET_NAMES = [
    "axial_A_to_T_fatSupressed_dicoms-20260214T164130Z-1-001/axial_A_to_T_fatSupressed_dicoms",
    "AXIAL fs PD FSE 256x256 GE MEDICAL SYSTEMS Optima MR360 CLINICA SANTA ANA hombroDerecho",
    "AXIAL-P5SE1 Clinica Especialistas Reina Virgen Maria QUIBDO HombroDer pd_tse_fs_tra_320 SIEMENS",
    "AXIAL(creo)ePDW_SPAIR IZQ SENSE 512x512 Philips Healthcare PHILIPS-EK2RD17 Ingenia Dpto Radiologia SOMA hombroIzquierdo",
    "dp_tse_cor_320_fs_8",
    "pd_tse_fs_tra_320_fov150_4",
    "t1_tse_fs_tra_320_fov150_5",
    "t1_tse_fs_tra_320_fov150_11",
]


def run_for_dataset(dataset_name: str) -> None:
    """Run segment_sam_propagation.main() for a single dataset.

    El modelo SAM y el predictor ya están cargados dentro de
    segment_sam_propagation; aquí solo cambiamos las rutas globales
    data_dir y output_dir antes de llamar a main().
    """
    in_dir = os.path.join(IN_ROOT, dataset_name)
    out_dir = os.path.join(OUT_ROOT, dataset_name)

    print("\n" + "=" * 80)
    print(f"🚀 Procesando dataset:\n   IN : {in_dir}\n   OUT: {out_dir}")
    print("=" * 80)

    # Guardar valores previos por si el usuario quiere reutilizar el módulo
    old_data_dir = ssp.data_dir
    old_output_dir = ssp.output_dir

    # Configurar rutas para esta ejecución
    ssp.data_dir = in_dir
    ssp.output_dir = out_dir
    os.makedirs(ssp.output_dir, exist_ok=True)

    try:
        ssp.main()
    finally:
        # Restaurar valores anteriores
        ssp.data_dir = old_data_dir
        ssp.output_dir = old_output_dir


def main() -> None:
    # Señalizar a las funciones de segmentación que estamos en modo batch
    # (sin GUI) para que usen el punto automático.
    os.environ["SAM_BATCH_MODE"] = "1"

    for name in DATASET_NAMES:
        run_for_dataset(name)


if __name__ == "__main__":
    main()
