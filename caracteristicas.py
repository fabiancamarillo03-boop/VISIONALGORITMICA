from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURACIÓN DEL PROYECTO
# ==========================================

PROYECTO = Path(__file__).resolve().parent.parent

BASE_DATOS = PROYECTO / "data" / "Task01_BrainTumour"

CARPETA_IMAGENES = BASE_DATOS / "imagesTr"
CARPETA_MASCARAS = BASE_DATOS / "labelsTr"

CARPETA_RESULTADOS = PROYECTO / "results"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

ARCHIVO_SALIDA = CARPETA_RESULTADOS / "caracteristicas.csv"

# Modalidades y etiquetas de la base de datos
MODALIDADES = {
    0: "FLAIR",
    1: "T1w",
    2: "T1_contraste",
    3: "T2w"
}

REGIONES = {
    1: "edema",
    2: "tumor_no_realzante",
    3: "tumor_realzante"
}


# ==========================================
# 2. EXTRAER ESTADÍSTICAS DE UNA REGIÓN
# ==========================================

def calcular_estadisticas(valores):

    if valores.size == 0:
        return {
            "media": np.nan,
            "desviacion": np.nan,
            "mediana": np.nan,
            "minimo": np.nan,
            "maximo": np.nan
        }

    return {
        "media": float(np.mean(valores)),
        "desviacion": float(np.std(valores)),
        "mediana": float(np.median(valores)),
        "minimo": float(np.min(valores)),
        "maximo": float(np.max(valores))
    }


# ==========================================
# 3. PROCESAR UN PACIENTE
# ==========================================

def procesar_paciente(ruta_imagen, ruta_mascara):

    # Leer archivos NIfTI
    imagen_sitk = sitk.ReadImage(str(ruta_imagen))
    mascara_sitk = sitk.ReadImage(str(ruta_mascara))

    imagen = sitk.GetArrayFromImage(imagen_sitk)
    mascara = sitk.GetArrayFromImage(mascara_sitk)

    # Comprobar dimensiones
    if imagen.ndim != 4 or imagen.shape[0] != 4:
        raise ValueError(
            f"Se esperaban 4 modalidades. Forma encontrada: {imagen.shape}"
        )

    if mascara.ndim != 3 or imagen.shape[1:] != mascara.shape:
        raise ValueError(
            "Las dimensiones de la resonancia y la máscara no coinciden."
        )

    # Volumen físico de un vóxel
    espaciado = imagen_sitk.GetSpacing()
    volumen_voxel = float(np.prod(espaciado[:3]))

    # Identificador del paciente
    paciente = ruta_imagen.name.replace(".nii.gz", "")

    resultados = {
        "paciente": paciente
    }

    # ======================================
    # 4. VOLUMEN DE LAS REGIONES
    # ======================================

    for etiqueta, nombre_region in REGIONES.items():

        region = (mascara == etiqueta)
        cantidad_voxeles = int(np.count_nonzero(region))

        volumen_mm3 = cantidad_voxeles * volumen_voxel

        resultados[f"{nombre_region}_voxeles"] = cantidad_voxeles
        resultados[f"{nombre_region}_volumen_mm3"] = volumen_mm3

    # Volumen total de las regiones etiquetadas
    tumor_total = (mascara > 0)

    cantidad_total = int(np.count_nonzero(tumor_total))

    resultados["total_voxeles_etiquetados"] = cantidad_total
    resultados["volumen_total_mm3"] = (
        cantidad_total * volumen_voxel
    )

    # ======================================
    # 5. ESTADÍSTICAS DE LAS MODALIDADES
    # ======================================

    for indice, nombre_modalidad in MODALIDADES.items():

        volumen_modalidad = imagen[indice]

        for etiqueta, nombre_region in REGIONES.items():

            region = (mascara == etiqueta)

            valores = volumen_modalidad[region]

            estadisticas = calcular_estadisticas(valores)

            for nombre_estadistica, valor in estadisticas.items():

                columna = (
                    f"{nombre_modalidad}_"
                    f"{nombre_region}_"
                    f"{nombre_estadistica}"
                )

                resultados[columna] = valor

    return resultados


# ==========================================
# 6. PROCESAR TODA LA BASE DE ENTRENAMIENTO
# ==========================================

def main():

    print("======================================")
    print(" EXTRACCIÓN DE CARACTERÍSTICAS")
    print(" MEDICAL SEGMENTATION DECATHLON")
    print("======================================")

    if not CARPETA_IMAGENES.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {CARPETA_IMAGENES}"
        )

    if not CARPETA_MASCARAS.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {CARPETA_MASCARAS}"
        )

    # Buscar imágenes reales, ignorando archivos auxiliares
    imagenes = sorted(
        archivo for archivo in CARPETA_IMAGENES.glob("*.nii.gz")
        if not archivo.name.startswith("._")
    )

    mascaras = {
        archivo.name: archivo
        for archivo in CARPETA_MASCARAS.glob("*.nii.gz")
        if not archivo.name.startswith("._")
    }

    print(f"\nImágenes encontradas: {len(imagenes)}")
    print(f"Máscaras encontradas: {len(mascaras)}")

    resultados = []
    errores = []

    # Procesar cada paciente
    for numero, ruta_imagen in enumerate(imagenes, start=1):

        ruta_mascara = mascaras.get(ruta_imagen.name)

        if ruta_mascara is None:
            errores.append(ruta_imagen.name)
            print(
                f"[{numero}/{len(imagenes)}] "
                f"Sin máscara: {ruta_imagen.name}"
            )
            continue

        try:

            datos = procesar_paciente(
                ruta_imagen,
                ruta_mascara
            )

            resultados.append(datos)

            print(
                f"[{numero}/{len(imagenes)}] "
                f"Procesado: {ruta_imagen.name}"
            )

        except Exception as error:

            errores.append(ruta_imagen.name)

            print(
                f"[{numero}/{len(imagenes)}] "
                f"Error en {ruta_imagen.name}: {error}"
            )

    # Guardar resultados
    if resultados:

        tabla = pd.DataFrame(resultados)

        tabla.to_csv(
            ARCHIVO_SALIDA,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n======================================")
        print(" PROCESAMIENTO FINALIZADO")
        print("======================================")

        print(f"Pacientes procesados: {len(resultados)}")
        print(f"Pacientes con errores: {len(errores)}")
        print(f"Características obtenidas: {len(tabla.columns) - 1}")
        print(f"\nArchivo guardado en:\n{ARCHIVO_SALIDA}")

    else:
        print("\nNo se pudieron procesar pacientes.")

    if errores:
        print("\nArchivos que necesitan revisión:")
        for archivo in errores:
            print(archivo)


if __name__ == "__main__":
    main()
