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
    # Aplicar ventana de pulmón
    min_visible = window_center - window_width / 2
    max_visible = window_center + window_width / 2

    # Recortar valores fuera de la ventana
    array_recortado = np.clip(array_volumen, min_visible, max_visible)

    # Normalizar a 0-255
    array_normalizado = ((array_recortado - min_visible) / (max_visible - min_visible) * 255).astype(np.uint8)

    print(f"Rango original: [{array_volumen.min():.0f}, {array_volumen.max():.0f}] HU")
    print(f"Ventana aplicada: [{min_visible:.0f}, {max_visible:.0f}] HU")
    print(f"Rango normalizado: [{array_normalizado.min():.0f}, {array_normalizado.max():.0f}] (0-255)")

    return array_normalizado


def crear_surface_rendering(vtk_image, threshold=80):
    """
    Crea una reconstrucción de superficie usando Marching Cubes
    """
    print(f"Aplicando Marching Cubes con umbral: {threshold}")

    # Aplicar filtro de suavizado opcional
    smoother = vtk.vtkImageGaussianSmooth()
    smoother.SetInputData(vtk_image)
    smoother.SetStandardDeviations(1.0, 1.0, 1.0)
    smoother.SetRadiusFactors(1.0, 1.0, 1.0)

    # Marching Cubes para extraer superficie
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(smoother.GetOutputPort())
    marching_cubes.SetValue(0, threshold)  # Umbral para tejido pulmonar
    marching_cubes.ComputeNormalsOn()

    print("Extrayendo superficie con Marching Cubes...")

    # Suavizar la malla resultante
    smoother_mesh = vtk.vtkSmoothPolyDataFilter()
    smoother_mesh.SetInputConnection(marching_cubes.GetOutputPort())
    smoother_mesh.SetNumberOfIterations(30)
    smoother_mesh.SetRelaxationFactor(0.1)
    smoother_mesh.FeatureEdgeSmoothingOff()
    smoother_mesh.BoundarySmoothingOn()

    # Reducir número de polígonos (opcional, para mejor rendimiento)
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputConnection(smoother_mesh.GetOutputPort())
    decimator.SetTargetReduction(0.3)  # Reducir 30% de polígonos
    decimator.PreserveTopologyOn()

    # Calcular normales para mejor iluminación
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(decimator.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    normals.SplittingOff()

    # Mapper y actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Propiedades de material para realismo
    actor.GetProperty().SetColor(0.8, 0.7, 0.6)  # Color carne
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetSpecular(0.4)
    actor.GetProperty().SetSpecularPower(40)
    actor.GetProperty().SetOpacity(1.0)

    return actor


def crear_volume_rendering(vtk_image):
    """
    Crea una visualización de volume rendering
    """
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image)

    # FUNCIÓN DE TRANSFERENCIA OPTIMIZADA PARA PULMONES
    opacity_tf = vtk.vtkPiecewiseFunction()
    # Aire/background - completamente transparente
    opacity_tf.AddPoint(0, 0.0)
    opacity_tf.AddPoint(20, 0.0)
    # Tejido pulmonar - semi-transparente para ver estructuras internas
    opacity_tf.AddPoint(40, 0.1)
    opacity_tf.AddPoint(80, 0.3)
    # Paredes bronquiales/vasos - más opacos
    opacity_tf.AddPoint(120, 0.6)
    # Tejidos densos/patologías - muy visibles
    opacity_tf.AddPoint(180, 0.8)
    opacity_tf.AddPoint(220, 1.0)

    # FUNCIÓN DE COLOR OPTIMIZADA
    color_tf = vtk.vtkColorTransferFunction()
    # Aire/background - negro transparente
    color_tf.AddRGBPoint(0, 0.0, 0.0, 0.0)
    # Tejido pulmonar sano - azul claro
    color_tf.AddRGBPoint(60, 0.3, 0.6, 0.9)
    # Tejido pulmonar medio - azul medio
    color_tf.AddRGBPoint(100, 0.5, 0.7, 1.0)
    # Paredes bronquiales - beige/amarillento
    color_tf.AddRGBPoint(140, 0.9, 0.8, 0.6)
    # Tejidos densos/consolidaciones - blanco/rojizo
    color_tf.AddRGBPoint(180, 1.0, 0.7, 0.6)
    # Estructuras muy densas - blanco puro
    color_tf.AddRGBPoint(220, 1.0, 1.0, 1.0)

    # PROPIEDADES DEL VOLUMEN MEJORADAS
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_tf)
    volume_property.SetScalarOpacity(opacity_tf)

    # Configuración de iluminación para mejor realismo
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.4)  # Luz ambiental
    volume_property.SetDiffuse(0.6)  # Luz difusa
    volume_property.SetSpecular(0.2)  # Reflexión especular
    volume_property.SetSpecularPower(10.0)

    # Interpolación para mejor calidad
    volume_property.SetInterpolationTypeToLinear()

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume


def visualizar_pulmones_surface(carpeta_dicom, max_slices=80, use_surface_rendering=True, threshold=80):
    """
    Visualización con Surface Rendering o Volume Rendering
    """
    try:
        print(f"Cargando estudio pulmonar: {carpeta_dicom}")

        # Leer archivos DICOM
        archivos = os.listdir(carpeta_dicom)
        archivos_dicom = [f for f in archivos if f.endswith('.dcm')]
        archivos_dicom.sort()

        print(f"Total slices DICOM: {len(archivos_dicom)}")

        # Limitar slices
        if len(archivos_dicom) > max_slices:
            step = len(archivos_dicom) // max_slices
            archivos_dicom = archivos_dicom[::step]
            print(f"Usando {len(archivos_dicom)} slices para visualización")

        slices = []
        for i, archivo in enumerate(archivos_dicom[:max_slices]):
            ruta_completa = os.path.join(carpeta_dicom, archivo)
            try:
                imagen = itk.imread(ruta_completa)
                array_img = itk.array_from_image(imagen)

                if array_img.ndim == 3 and array_img.shape[0] == 1:
                    array_img = array_img[0]

                slices.append(array_img)
                print(f"Cargando slice {i + 1}/{len(archivos_dicom[:max_slices])}", end='\r')

            except Exception as e:
                continue

        if not slices:
            print("ERROR: No se pudieron cargar slices")
            return

        # Crear volumen 3D
        array_volumen = np.stack(slices, axis=0)

        print(f"\n=== NORMALIZACIÓN A 8-BITS ===")
        print(f"Dimensiones originales: {array_volumen.shape}")

        # NORMALIZAR A 0-255
        array_8bit = normalizar_a_8bits(array_volumen, window_center=-600, window_width=1500)

        # VISUALIZACIÓN CON DATOS NORMALIZADOS
        print("\nPreparando visualización...")

        # Convertir a VTK (usar VTK_UNSIGNED_CHAR para 8-bits)
        vtk_data = numpy_support.numpy_to_vtk(
            array_8bit.ravel(order='F'),
            array_type=vtk.VTK_UNSIGNED_CHAR  # 8-bits sin signo
        )

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(array_8bit.shape[2], array_8bit.shape[1], array_8bit.shape[0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        # CREAR VISUALIZACIÓN CON MÚLTIPLES VISTAS
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(1400, 900)

        if use_surface_rendering:
            render_window.SetWindowName("Visualización Pulmonar - Surface Rendering")
            rendering_type = "Surface Rendering"
        else:
            render_window.SetWindowName("Visualización Pulmonar - Volume Rendering")
            rendering_type = "Volume Rendering"

        # Viewports para 4 vistas
        viewports = [
            (0.0, 0.5, 0.5, 1.0),  # 3D - arriba izquierda
            (0.5, 0.5, 1.0, 1.0),  # Axial - arriba derecha
            (0.0, 0.0, 0.5, 0.5),  # Coronal - abajo izquierda
            (0.5, 0.0, 1.0, 0.5)  # Sagital - abajo derecha
        ]

        titulos = [
            f"3D - {rendering_type}",
            "Corte Axial",
            "Corte Coronal",
            "Corte Sagital"
        ]
        renderers = []

        for i, viewport in enumerate(viewports):
            renderer = vtk.vtkRenderer()
            renderer.SetViewport(viewport)
            renderer.SetBackground(0.1, 0.1, 0.1)
            render_window.AddRenderer(renderer)
            renderers.append(renderer)

            # Texto para cada vista
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(titulos[i])
            text_actor.GetTextProperty().SetFontSize(14)
            text_actor.GetTextProperty().SetColor(1, 1, 1)
            text_actor.SetPosition(10, 10)
            renderer.AddActor2D(text_actor)

        # 1. VISTA 3D PRINCIPAL (Surface Rendering o Volume Rendering)
        if use_surface_rendering:
            print("Generando Surface Rendering...")
            surface_actor = crear_surface_rendering(vtk_image, threshold=threshold)
            renderers[0].AddActor(surface_actor)
        else:
            print("Generando Volume Rendering...")
            volume = crear_volume_rendering(vtk_image)
            renderers[0].AddVolume(volume)

        # 2-4. VISTAS 2D CON DATOS NORMALIZADOS
        slice_positions = [
            array_8bit.shape[0] // 2,  # Axial
            array_8bit.shape[1] // 2,  # Coronal
            array_8bit.shape[2] // 2  # Sagital
        ]

        for i in range(1, 4):
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(vtk_image)
            reslice.SetOutputDimensionality(2)

            # Configurar orientación para cada plano
            if i == 1:  # Axial
                reslice.SetResliceAxesDirectionCosines([1, 0, 0, 0, 1, 0, 0, 0, 1])
                reslice.SetResliceAxesOrigin([0, 0, slice_positions[0]])
            elif i == 2:  # Coronal
                reslice.SetResliceAxesDirectionCosines([1, 0, 0, 0, 0, 1, 0, 1, 0])
                reslice.SetResliceAxesOrigin([0, slice_positions[1], 0])
            else:  # Sagital
                reslice.SetResliceAxesDirectionCosines([0, 0, 1, 0, 1, 0, 1, 0, 0])
                reslice.SetResliceAxesOrigin([slice_positions[2], 0, 0])

            # Para datos 8-bits, usar ventana completa (0-255)
            mapper = vtk.vtkImageMapper()
            mapper.SetInputConnection(reslice.GetOutputPort())
            mapper.SetColorWindow(255)  # Rango completo
            mapper.SetColorLevel(128)  # Centro del rango

            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            renderers[i].AddActor(actor)

        # INTERACTOR
        render_interactor = vtk.vtkRenderWindowInteractor()
        render_interactor.SetRenderWindow(render_window)

        # Configurar cámara 3D
        renderers[0].GetActiveCamera().SetFocalPoint(0, 0, 0)
        renderers[0].GetActiveCamera().SetPosition(0, -1, 0)
        renderers[0].GetActiveCamera().SetViewUp(0, 0, 1)
        renderers[0].ResetCamera()

        # TEXTO INFORMATIVO
        info_text = vtk.vtkTextActor()
        info_text.SetInput(
            f"Tipo: {rendering_type} | Slices: {len(slices)} | "
            f"Umbral: {threshold} | Ventana: Pulmón [W:1500 L:-600]"
        )
        info_text.GetTextProperty().SetFontSize(16)
        info_text.GetTextProperty().SetColor(1, 1, 1)
        info_text.SetPosition(10, 860)
        renderers[0].AddActor2D(info_text)

        print(f"\n=== {rendering_type.upper()} LISTO ===")
        print(f"Técnica: {rendering_type}")
        if use_surface_rendering:
            print(f"Umbral de Marching Cubes: {threshold}")
        print("Controles:")
        print("- Rotar vista 3D con mouse")
        print("- R: Reset cámaras")
        print("- Q: Salir")

        def key_press_callback(obj, event):
            key = obj.GetKeySym()
            if key == 'r' or key == 'R':
                for renderer in renderers:
                    renderer.ResetCamera()
                # Configurar cámara 3D específica
                renderers[0].GetActiveCamera().SetFocalPoint(0, 0, 0)
                renderers[0].GetActiveCamera().SetPosition(0, -1, 0)
                renderers[0].GetActiveCamera().SetViewUp(0, 0, 1)
                render_window.Render()
            elif key == 'q' or key == 'Q':
                render_window.Finalize()
                render_interactor.TerminateApp()

        render_interactor.AddObserver("KeyPressEvent", key_press_callback)

        render_window.Render()
        render_interactor.Start()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def menu_principal():
    """
    Menú para seleccionar tipo de visualización
    """
    print("=" * 60)
    print("VISUALIZACIÓN 3D DE PULMONES")
    print("=" * 60)
    print("1. Surface Rendering (Reconstrucción de Superficie)")
    print("2. Volume Rendering (Renderizado Volumétrico)")
    print("3. Probar diferentes umbrales (Surface Rendering)")

    opcion = input("\nSeleccione una opción (1-3): ").strip()

    carpeta_dicom = "/home/isaac/Descargas/Covid Scans/Covid Scans/Subject (1)/98.12.2"

    if opcion == "1":
        print("\nUsando Surface Rendering...")
        visualizar_pulmones_surface(carpeta_dicom, use_surface_rendering=True, threshold=80)
    elif opcion == "2":
        print("\nUsando Volume Rendering...")
        visualizar_pulmones_surface(carpeta_dicom, use_surface_rendering=False)
    elif opcion == "3":
        print("\nProbando diferentes umbrales para Surface Rendering...")
        umbrales = [60, 80, 100, 120]
        for umbral in umbrales:
            print(f"\n--- Probando umbral: {umbral} ---")
            visualizar_pulmones_surface(carpeta_dicom, use_surface_rendering=True, threshold=umbral)
    else:
        print("Opción no válida. Usando Surface Rendering por defecto.")
        visualizar_pulmones_surface(carpeta_dicom, use_surface_rendering=True, threshold=80)


# EJECUTAR
if __name__ == "__main__":
    menu_principal()