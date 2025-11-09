import itk
import vtk
from vtk.util import numpy_support
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

def normalizar_a_8bits(array_volumen, window_center=-600, window_width=1500):
    """
    Normaliza el volumen DICOM al rango 0-255 usando ventana de pulmón
    """
    min_visible = window_center - window_width / 2
    max_visible = window_center + window_width / 2
    array_recortado = np.clip(array_volumen, min_visible, max_visible)
    array_normalizado = ((array_recortado - min_visible) / (max_visible - min_visible) * 255).astype(np.uint8)

    print(f"Rango original: [{array_volumen.min():.0f}, {array_volumen.max():.0f}] HU")
    print(f"Ventana aplicada: [{min_visible:.0f}, {max_visible:.0f}] HU")
    print(f"Rango normalizado: [{array_normalizado.min():.0f}, {array_normalizado.max():.0f}] (0-255)")
    return array_normalizado

def visualizar_pulmones_normalizado(carpeta_dicom, max_slices=80):
    try:
        print(f"Cargando estudio DICOM desde: {carpeta_dicom}")

        archivos = sorted([f for f in os.listdir(carpeta_dicom) if f.endswith('.dcm')])
        if not archivos:
            print("No se encontraron archivos DICOM.")
            return

        if len(archivos) > max_slices:
            step = len(archivos) // max_slices
            archivos = archivos[::step]
            print(f"Usando {len(archivos)} slices de {len(os.listdir(carpeta_dicom))}")

        slices = []
        # Obtener el espaciado correcto desde los metadatos DICOM
        try:
            spacing_itk = img.GetSpacing()  # (z, y, x)
            print("Espaciado ITK (z, y, x):", img.GetSpacing())
            spacing_vtk = (spacing_itk[2], spacing_itk[1], spacing_itk[0])  # Reordenar a (x, y, z)
        except Exception:
            spacing_vtk = (1.0, 1.0, 1.0)

        spacing = spacing_vtk
        print(f"Espaciado aplicado a VTK (x, y, z): {spacing}")

        for archivo in archivos:
            ruta = os.path.join(carpeta_dicom, archivo)
            img = itk.imread(ruta)
            array = itk.array_from_image(img)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            slices.append(array)
            spacing = img.GetSpacing()

        array_volumen = np.stack(slices, axis=0)
        print(f"Dimensiones del volumen: {array_volumen.shape}")
        print(f"Espaciado detectado: {spacing}")

        array_8bit = normalizar_a_8bits(array_volumen, window_center=-600, window_width=1500)

        vtk_data = numpy_support.numpy_to_vtk(array_8bit.ravel(order='F'), array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(array_8bit.shape[2], array_8bit.shape[1], array_8bit.shape[0])
        vtk_image.SetSpacing(spacing)
        vtk_image.GetPointData().SetScalars(vtk_data)

        # --- Configuración de Render ---
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(1400, 900)
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.05, 0.05, 0.05)
        render_window.AddRenderer(renderer)

        # --- Volume Rendering con parámetros mejorados ---
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)
        volume_mapper.SetBlendModeToComposite()
        volume_mapper.SetSampleDistance(0.2)
        volume_mapper.SetAutoAdjustSampleDistances(True)

        # --- Funciones de opacidad y color ajustadas ---
        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(0, 0.0)
        opacity_tf.AddPoint(30, 0.0)
        opacity_tf.AddPoint(60, 0.05)
        opacity_tf.AddPoint(100, 0.2)
        opacity_tf.AddPoint(150, 0.5)
        opacity_tf.AddPoint(220, 1.0)

        color_tf = vtk.vtkColorTransferFunction()
        color_tf.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(60, 0.3, 0.6, 0.9)
        color_tf.AddRGBPoint(100, 0.6, 0.8, 1.0)
        color_tf.AddRGBPoint(150, 1.0, 0.8, 0.6)
        color_tf.AddRGBPoint(220, 1.0, 1.0, 1.0)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_tf)
        volume_property.SetScalarOpacity(opacity_tf)
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.3)
        volume_property.SetDiffuse(0.7)
        volume_property.SetSpecular(0.3)
        volume_property.SetSpecularPower(20.0)
        volume_property.SetInterpolationTypeToLinear()

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        renderer.AddVolume(volume)

        # --- Cámara e interacción ---
        renderer.GetActiveCamera().Elevation(30)
        renderer.GetActiveCamera().Azimuth(45)
        renderer.ResetCamera()

        info = vtk.vtkTextActor()
        info.SetInput("Render Volumétrico - Pulmones (Normalizado 0-255)")
        info.GetTextProperty().SetFontSize(16)
        info.GetTextProperty().SetColor(1, 1, 1)
        info.SetPosition(10, 870)
        renderer.AddActor2D(info)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        def key_callback(obj, event):
            k = obj.GetKeySym()
            if k.lower() == 'r':
                renderer.ResetCamera()
                renderer.GetActiveCamera().Elevation(30)
                renderer.GetActiveCamera().Azimuth(45)
                render_window.Render()

        interactor.AddObserver("KeyPressEvent", key_callback)
        render_window.Render()
        interactor.Start()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    carpeta_dicom = "/home/isaac/Descargas/Covid Scans/Covid Scans/Subject (1)/98.12.2"
    visualizar_pulmones_normalizado(carpeta_dicom)
    print("Espaciado ITK (z, y, x):", img.GetSpacing())

