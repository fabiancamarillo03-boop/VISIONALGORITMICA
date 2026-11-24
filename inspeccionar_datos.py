from pathlib import Path
import SimpleITK as sitk
import numpy as np

# Localizar la carpeta principal del proyecto
PROYECTO = Path(__file__).resolve().parent.parent

# Rutas de nuestra base de datos
BASE_DATOS = PROYECTO / "data" / "Task01_BrainTumour"

RUTA_IMAGEN = BASE_DATOS / "imagesTr" / "BRATS_001.nii.gz"
RUTA_MASCARA = BASE_DATOS / "labelsTr" / "BRATS_001.nii.gz"


def inspeccionar_archivo(ruta, nombre):
    print(f"\n--- {nombre} ---")

    if not ruta.exists():
        print(f"No se encontró el archivo: {ruta}")
        return None

    imagen = sitk.ReadImage(str(ruta))
    arreglo = sitk.GetArrayFromImage(imagen)

    print("Dimensiones:", imagen.GetDimension())
    print("Tamaño:", imagen.GetSize())
    print("Forma del arreglo:", arreglo.shape)
    print("Tipo de píxel:", imagen.GetPixelIDTypeAsString())
    print("Espaciado:", imagen.GetSpacing())

    return arreglo


def main():
    print("INSPECCIÓN DE LA BASE DE DATOS BRATS")
    
    imagen = inspeccionar_archivo(
        RUTA_IMAGEN, "RESONANCIA MAGNÉTICA"
    )

    mascara = inspeccionar_archivo(
        RUTA_MASCARA, "MÁSCARA DE SEGMENTACIÓN"
    )

    if mascara is not None:
        etiquetas = np.unique(mascara)
        print("\nEtiquetas encontradas en la máscara:")
        print(etiquetas)

    if imagen is not None and mascara is not None:
        print("\n¡Lectura de ambos archivos completada!")


if __name__ == "__main__":
    main()
