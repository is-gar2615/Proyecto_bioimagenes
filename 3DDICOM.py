import itk
import vtk
from vtk.util import numpy_support
import numpy as np
import os
# import warnings

# warnings.filterwarnings("ignore")


def reducir_resolucion_simple(array_img, nuevo_tamano=(256, 256)):
    """
    Reducción simple de resolución sin scikit-image
    """
    original_h, original_w = array_img.shape
    nuevo_h, nuevo_w = nuevo_tamano

    # Calcular factores de escala
    scale_h = original_h / nuevo_h
    scale_w = original_w / nuevo_w

    # Crear nueva imagen
    nueva_imagen = np.zeros((nuevo_h, nuevo_w), dtype=array_img.dtype)

    # Reducción simple por promediado
    for i in range(nuevo_h):
        for j in range(nuevo_w):
            start_h = int(i * scale_h)
            end_h = int((i + 1) * scale_h)
            start_w = int(j * scale_w)
            end_w = int((j + 1) * scale_w)

            # Tomar bloque y promediar
            bloque = array_img[start_h:end_h, start_w:end_w]
            if bloque.size > 0:
                nueva_imagen[i, j] = np.mean(bloque)

    return nueva_imagen


def visualizar_volumen_optimizado(carpeta_dicom, reducir_resolucion=True, max_slices=50):
    """
    Versión optimizada para volúmenes grandes - SIN scikit-image
    """
    try:
        print(f"Cargando {carpeta_dicom}")

        # Leer archivos manualmente
        archivos = os.listdir(carpeta_dicom)
        archivos_dicom = [f for f in archivos if f.endswith('.dcm')]
        archivos_dicom.sort()

        print(f"Total archivos DICOM: {len(archivos_dicom)}")

        # Limitar número de slices
        if len(archivos_dicom) > max_slices and reducir_resolucion:
            step = len(archivos_dicom) // max_slices
            archivos_dicom = archivos_dicom[::step]
            print(f"Usando {len(archivos_dicom)} slices (reducido)")

        slices = []
        for i, archivo in enumerate(archivos_dicom[:max_slices]):
            ruta_completa = os.path.join(carpeta_dicom, archivo)
            try:
                imagen = itk.imread(ruta_completa)
                array_img = itk.array_from_image(imagen)

                # Remover dimensión unitaria si existe (1, 512, 512) -> (512, 512)
                if array_img.shape[0] == 1:
                    array_img = array_img[0]

                # Reducir resolución si es necesario (sin scikit-image)
                if reducir_resolucion and array_img.shape[0] > 256:
                    # Método simple: tomar cada 2do pixel
                    array_img = array_img[::2, ::2]

                slices.append(array_img)
                print(f"Procesado slice {i + 1}/{len(archivos_dicom[:max_slices])}", end='\r')

            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")
                continue

        if not slices:
            print("ERROR: No se pudieron leer slices")
            return

        # Crear volumen 3D
        array_volumen = np.stack(slices, axis=0)

        print(f"\nVolumen final: {array_volumen.shape}")
        print(f"Rango: [{array_volumen.min():.1f}, {array_volumen.max():.1f}]")

        # Convertir a tipo de datos más eficiente
        if array_volumen.dtype == np.int32:
            array_volumen = array_volumen.astype(np.int16)

        # VISUALIZACIÓN OPTIMIZADA
        print("Preparando visualización VTK...")

        # Convertir a VTK
        vtk_data = numpy_support.numpy_to_vtk(
            array_volumen.ravel(order='F'),
            array_type=vtk.VTK_SHORT
        )

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(array_volumen.shape[2], array_volumen.shape[1], array_volumen.shape[0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        # Usar un mapper más simple
        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        # Propiedades simples
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()

        # Función de transferencia para CT (valores típicos)
        min_val = array_volumen.min()
        max_val = array_volumen.max()

        # Ajustar para rango típico de CT (-1000 a 3000)
        window_center = 50  # Nivel típico para tejidos blandos
        window_width = 400  # Ancho típico

        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(min_val, 0.0)
        opacity_transfer_function.AddPoint(window_center - window_width / 2, 0.0)
        opacity_transfer_function.AddPoint(window_center, 0.3)
        opacity_transfer_function.AddPoint(window_center + window_width / 2, 0.0)
        opacity_transfer_function.AddPoint(max_val, 0.0)

        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(window_center - window_width / 2, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(window_center, 0.8, 0.8, 0.8)
        color_transfer_function.AddRGBPoint(window_center + window_width / 2, 1.0, 1.0, 1.0)
        color_transfer_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0)

        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.1)
        renderer.AddVolume(volume)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)
        render_window.SetWindowName(f"Volumen CT - {array_volumen.shape}")

        render_interactor = vtk.vtkRenderWindowInteractor()
        render_interactor.SetRenderWindow(render_window)

        renderer.ResetCamera()

        print("¡Visualización lista! Rotar con mouse, Q para salir")
        render_window.Render()
        render_interactor.Start()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


# Versión ULTRA liviana para probar primero
def visualizar_solo_few_slices(carpeta_dicom, num_slices=10):
    """
    Visualiza solo unos pocos slices para prueba rápida
    """
    try:
        print(f"Cargando primeros {num_slices} slices...")

        archivos = os.listdir(carpeta_dicom)
        archivos_dicom = [f for f in archivos if f.endswith('.dcm')]
        archivos_dicom.sort()

        slices = []
        for i, archivo in enumerate(archivos_dicom[:num_slices]):
            ruta_completa = os.path.join(carpeta_dicom, archivo)
            imagen = itk.imread(ruta_completa)
            array_img = itk.array_from_image(imagen)

            # Remover dimensión unitaria
            if array_img.shape[0] == 1:
                array_img = array_img[0]

            # Reducción simple
            if array_img.shape[0] > 256:
                array_img = array_img[::2, ::2]

            slices.append(array_img)
            print(f"Slice {i + 1}/{num_slices} - {array_img.shape}")

        array_volumen = np.stack(slices, axis=0)
        print(f"Volumen: {array_volumen.shape}")

        # Visualización (código VTK igual...)
        vtk_data = numpy_support.numpy_to_vtk(
            array_volumen.ravel(order='F'),
            array_type=vtk.VTK_SHORT
        )

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(array_volumen.shape[2], array_volumen.shape[1], array_volumen.shape[0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()

        min_val, max_val = array_volumen.min(), array_volumen.max()

        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(min_val, 0.0)
        opacity_transfer_function.AddPoint(max_val * 0.3, 0.2)
        opacity_transfer_function.AddPoint(max_val * 0.6, 0.4)

        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(max_val * 0.5, 0.8, 0.8, 0.8)
        color_transfer_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0)

        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.1)
        renderer.AddVolume(volume)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)
        render_window.SetWindowName(f"Preview {num_slices} slices")

        render_interactor = vtk.vtkRenderWindowInteractor()
        render_interactor.SetRenderWindow(render_window)

        renderer.ResetCamera()

        print("¡Preview listo!")
        render_window.Render()
        render_interactor.Start()

    except Exception as e:
        print(f"ERROR: {e}")


# EJECUTAR - prueba en este orden:
if __name__ == "__main__":
    carpeta_dicom = "/home/isaac/Descargas/Covid Scans/Covid Scans/Subject (1)/98.12.2"

    # 1. PRIMERO prueba con solo 5 slices
    #print("=== PRUEBA CON 5 SLICES ===")
    #visualizar_solo_few_slices(carpeta_dicom, num_slices=5)

    # 2. Si funciona, prueba con más
    #print("=== PRUEBA CON 30 SLICES ===")
    #visualizar_volumen_optimizado(carpeta_dicom, reducir_resolucion=True, max_slices=30)

    # 3. Finalmente con más slices
    print("=== PRUEBA CON 150 SLICES ===")
    visualizar_volumen_optimizado(carpeta_dicom, reducir_resolucion=True, max_slices=150)

    #https://3dicomviewer.com/dicom-library/