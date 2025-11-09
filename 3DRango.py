import itk
import vtk
from vtk.util import numpy_support
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


def visualizar_pulmones_3d(carpeta_dicom, max_slices=100):
    """
    Visualización especializada para pulmones/COVID-19
    Aplica ventana de pulmón para mejor contraste
    """
    try:
        print(f"Cargando estudio pulmonar: {carpeta_dicom}")

        # Leer archivos DICOM
        archivos = os.listdir(carpeta_dicom)
        archivos_dicom = [f for f in archivos if f.endswith('.dcm')]
        archivos_dicom.sort()

        print(f"Total slices DICOM: {len(archivos_dicom)}")

        # Limitar slices para rendimiento
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

                # Remover dimensión unitaria
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

        print(f"\n=== INFORMACIÓN DEL VOLUMEN ===")
        print(f"Dimensiones: {array_volumen.shape}")
        print(f"Rango HU: [{array_volumen.min():.0f}, {array_volumen.max():.0f}]")

        # PARÁMETROS ESPECÍFICOS PARA PULMÓN
        window_center = -600  # Típico para pulmón
        window_width = 1500  # Ancho para ver detalles pulmonares
        min_visible = window_center - window_width / 2
        max_visible = window_center + window_width / 2

        print(f"Ventana PULMÓN: Centro={window_center}, Ancho={window_width}")
        print(f"Rango visible: [{min_visible:.0f}, {max_visible:.0f}] HU")

        # Convertir a tipo de datos eficiente
        if array_volumen.dtype == np.int32:
            array_volumen = array_volumen.astype(np.int16)

        # VISUALIZACIÓN 3D ESPECIALIZADA PARA PULMÓN
        print("\nPreparando visualización 3D de pulmones...")

        # Convertir a VTK
        vtk_data = numpy_support.numpy_to_vtk(
            array_volumen.ravel(order='F'),
            array_type=vtk.VTK_SHORT
        )

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(array_volumen.shape[2], array_volumen.shape[1], array_volumen.shape[0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        # Mapper para volume rendering
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        # FUNCIONES DE TRANSFERENCIA PARA PULMÓN
        opacity_transfer_function = vtk.vtkPiecewiseFunction()

        # Configurar opacidad para resaltar tejido pulmonar
        # Aire: ~-1000 HU (transparente)
        # Tejido pulmonar sano: ~-800 a -600 HU
        # Tejido afectado/consolidado: > -500 HU
        # Pared torácica: > -200 HU

        opacity_transfer_function.AddPoint(-1024, 0.0)  # Aire completamente transparente
        opacity_transfer_function.AddPoint(-900, 0.0)  #
        opacity_transfer_function.AddPoint(-800, 0.1)  # Inicio tejido pulmonar
        opacity_transfer_function.AddPoint(-700, 0.3)  # Tejido pulmonar
        opacity_transfer_function.AddPoint(-600, 0.5)  # Máxima visibilidad pulmonar
        opacity_transfer_function.AddPoint(-500, 0.8)  # Tejido consolidado (COVID)
        opacity_transfer_function.AddPoint(-200, 1.0)  # Pared torácica
        opacity_transfer_function.AddPoint(100, 1.0)  # Tejidos densos
        opacity_transfer_function.AddPoint(1000, 1.0)  # Hueso

        # Función de color - escala de grises médica con énfasis en pulmón
        color_transfer_function = vtk.vtkColorTransferFunction()

        color_transfer_function.AddRGBPoint(-1024, 0.0, 0.0, 0.0)  # Negro - aire
        color_transfer_function.AddRGBPoint(-900, 0.1, 0.1, 0.3)  # Azul muy oscuro
        color_transfer_function.AddRGBPoint(-800, 0.2, 0.3, 0.6)  # Azul
        color_transfer_function.AddRGBPoint(-700, 0.4, 0.5, 0.8)  # Azul claro
        color_transfer_function.AddRGBPoint(-600, 0.8, 0.8, 0.9)  # Gris azulado claro
        color_transfer_function.AddRGBPoint(-500, 0.9, 0.7, 0.3)  # Amarillo - posible afectación
        color_transfer_function.AddRGBPoint(-200, 0.9, 0.5, 0.2)  # Naranja - consolidación
        color_transfer_function.AddRGBPoint(100, 0.8, 0.8, 0.8)  # Gris - tejidos blandos
        color_transfer_function.AddRGBPoint(1000, 1.0, 1.0, 1.0)  # Blanco - hueso

        # Propiedades del volumen
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.4)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.2)

        # Crear volumen
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # Renderer con fondo profesional
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.05, 0.05, 0.1)  # Azul oscuro profesional
        renderer.AddVolume(volume)

        # Luz adicional para mejor contraste
        light = vtk.vtkLight()
        light.SetPosition(0, 0, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(1, 1, 1)
        light.SetIntensity(1.0)
        renderer.AddLight(light)

        # Ventana de renderizado
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1200, 900)
        render_window.SetWindowName("Visualización 3D de Pulmones - Estudio COVID-19")

        # Interactor
        render_interactor = vtk.vtkRenderWindowInteractor()
        render_interactor.SetRenderWindow(render_window)

        # Configurar cámara para vista anatómica
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.Elevation(30)  # Vista desde arriba
        camera.Azimuth(45)  # Vista oblicua
        camera.Dolly(1.5)  # Un poco de zoom

        # Añadir texto informativo
        def add_text(renderer, text, position):
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(text)
            text_actor.GetTextProperty().SetFontSize(16)
            text_actor.GetTextProperty().SetColor(1, 1, 1)
            text_actor.SetPosition(position[0], position[1])
            renderer.AddActor2D(text_actor)

        add_text(renderer, "VISUALIZACIÓN 3D PULMONAR", (10, 850))
        add_text(renderer, f"Ventana: Pulmón [W:{window_width} L:{window_center}]", (10, 820))
        add_text(renderer, f"Slices: {len(slices)} | Resolución: {array_volumen.shape}", (10, 790))

        print("\n=== VISUALIZACIÓN LISTA ===")
        print("Vista 3D de pulmones con ventana específica para COVID-19")
        print("Controles:")
        print("- Click + arrastrar: Rotar vista 3D")
        print("- Rueda mouse: Zoom")
        print("- R: Reset cámara")
        print("- C: Cambiar vista coronal")
        print("- A: Cambiar vista axial")
        print("- S: Cambiar vista sagital")
        print("- Q: Salir")

        # Función para cambiar vistas
        def key_press_callback(obj, event):
            key = obj.GetKeySym()
            camera = renderer.GetActiveCamera()

            if key == 'c' or key == 'C':
                camera.SetPosition(0, -1, 0)
                camera.SetViewUp(0, 0, 1)
                print("Vista: Coronal")
            elif key == 'a' or key == 'A':
                camera.SetPosition(0, 0, 1)
                camera.SetViewUp(0, 1, 0)
                print("Vista: Axial")
            elif key == 's' or key == 'S':
                camera.SetPosition(1, 0, 0)
                camera.SetViewUp(0, 0, 1)
                print("Vista: Sagital")
            elif key == 'r' or key == 'R':
                renderer.ResetCamera()
                print("Cámara reseteada")

            render_window.Render()

        render_interactor.AddObserver("KeyPressEvent", key_press_callback)

        # Renderizar
        render_window.Render()
        print("¡Visualización 3D activa!")
        render_interactor.Start()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


# Versión alternativa con énfasis en opacidades para COVID
def visualizar_pulmones_covid_enfasis(carpeta_dicom, max_slices=80):
    """
    Enfasis en opacidades típicas de COVID-19 (vidrio esmerilado, consolidaciones)
    """
    # Similar al anterior pero con opacidades ajustadas para patología COVID
    visualizar_pulmones_3d(carpeta_dicom, max_slices)


# EJECUTAR
if __name__ == "__main__":
    carpeta_dicom = "/home/isaac/Descargas/Covid Scans/Covid Scans/Subject (1)/98.12.2"

    print("=" * 60)
    print("VISUALIZADOR 3D DE PULMONES - ESTUDIO COVID-19")
    print("=" * 60)

    # Usar la visualización especializada para pulmones
    visualizar_pulmones_3d(carpeta_dicom, max_slices=80)