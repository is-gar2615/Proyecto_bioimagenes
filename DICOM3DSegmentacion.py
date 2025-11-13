import itk
import vtk
import numpy as np
import os
import glob
import pydicom
from vtk.util import numpy_support
import matplotlib.pyplot as plt


class DICOM3DViewer:
    def __init__(self, dicom_directory):
        self.dicom_directory = dicom_directory
        self.image = None
        self.array = None

    def find_dicom_files(self, max_slices=None):
        """Encuentra y verifica archivos DICOM de manera robusta"""
        print(f"Buscando archivos DICOM en: {self.dicom_directory}")

        # Lista todas las extensiones posibles de DICOM
        possible_extensions = ['*.dcm', '*.DCM', '*.dic', '*.DIC', '*.dicom', '*.DICOM', '*']
        dicom_files = []

        for ext in possible_extensions:
            files = glob.glob(os.path.join(self.dicom_directory, ext))
            # Filtrar solo archivos (no directorios)
            files = [f for f in files if os.path.isfile(f)]
            dicom_files.extend(files)

        # Si hay muchos archivos, filtrar por tamaño típico de DICOM (> 1KB)
        if len(dicom_files) > 100:
            dicom_files = [f for f in dicom_files if os.path.getsize(f) > 1024]

        # Verificar cuáles son realmente archivos DICOM
        verified_dicom_files = []
        for file_path in dicom_files:
            try:
                pydicom.dcmread(file_path, stop_before_pixels=True)
                verified_dicom_files.append(file_path)
                print(f"✓ {os.path.basename(file_path)} - DICOM válido")
            except:
                print(f"✗ {os.path.basename(file_path)} - No es DICOM")
                continue

        if not verified_dicom_files:
            raise ValueError("No se encontraron archivos DICOM válidos en el directorio")

        print(f"Se encontraron {len(verified_dicom_files)} archivos DICOM válidos")

        # Ordenar los archivos (importante para series DICOM)
        verified_dicom_files.sort()

        # Limitar número de slices si se especifica
        if max_slices and max_slices < len(verified_dicom_files):
            verified_dicom_files = verified_dicom_files[:max_slices]
            print(f"Usando {max_slices} slices de {len(verified_dicom_files)} disponibles")

        return verified_dicom_files

    def load_dicom_series(self, max_slices=None):
        """Carga la serie DICOM con manejo robusto de errores"""
        print("Cargando serie DICOM...")

        # Encontrar archivos DICOM
        dicom_files = self.find_dicom_files(max_slices)

        if not dicom_files:
            raise ValueError("No se pudieron encontrar archivos DICOM")

        # Usar el método manual que funciona mejor
        self._load_slices_manual(dicom_files)

        print(f"✓ Dimensión de la imagen: {self.array.shape}")
        print(f"✓ Rango de valores: {np.min(self.array):.2f} a {np.max(self.array):.2f}")

    def _load_slices_manual(self, dicom_files):
        """Método alternativo para cargar slices manualmente - CORREGIDO"""
        print("Cargando slices manualmente...")

        slices = []
        for i, file_path in enumerate(dicom_files):
            try:
                # OMITIR EL PRIMER SLICE (índice 0) - información del estudio
                if i == 0:
                    print(f"  Saltando slice {i + 1}/{len(dicom_files)} - información del estudio")
                    continue

                # Leer cada slice individualmente
                image = itk.imread(file_path)
                array = itk.array_from_image(image)

                # CORRECIÓN: Aplanar dimensiones innecesarias
                if array.ndim == 3:
                    # Si la forma es (1, 512, 512), tomar solo el primer canal
                    array = array[0]  # Esto convierte (1, 512, 512) a (512, 512)

                slices.append(array)
                print(f"  Slice {i + 1}/{len(dicom_files)} cargado - forma: {array.shape}")
            except Exception as e:
                print(f"  Error cargando slice {i + 1}: {e}")
                continue

        if not slices:
            raise ValueError("No se pudo cargar ningún slice")

        # Combinar todos los slices en un array 3D
        self.array = np.stack(slices, axis=0)
        print(f"Array 3D final - forma: {self.array.shape}")
        print(f"Slices cargados: {len(slices)} (se omitió el slice 0 con información del estudio)")

        # Convertir de vuelta a imagen ITK
        self.image = itk.image_from_array(self.array)

    def show_slice_preview(self, slice_index=None):
        """Muestra una preview 2D de slices"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print(f"Mostrando preview - forma del array: {self.array.shape}")

        # Mostrar múltiples slices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # Seleccionar slices equidistantes
        total_slices = self.array.shape[0]
        preview_slices = [
            0,
            total_slices // 4,
            total_slices // 2,
            3 * total_slices // 4,
            total_slices - 1
        ]

        for i, slice_idx in enumerate(preview_slices[:len(axes) - 1]):
            if slice_idx < total_slices:
                im = axes[i].imshow(self.array[slice_idx], cmap='gray')
                axes[i].set_title(f'Slice {slice_idx}/{total_slices}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046)

        # Histograma en el último subplot
        axes[-1].hist(self.array.ravel(), bins=50, alpha=0.7)
        axes[-1].set_title('Histograma de intensidades')
        axes[-1].set_xlabel('Intensidad')
        axes[-1].set_ylabel('Frecuencia')

        plt.tight_layout()
        plt.show()

    def volume_rendering_simple(self):
        """Renderizado volumétrico simplificado"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print("Preparando renderizado volumétrico simplificado...")

        # Asegurarnos de que el array tenga la forma correcta para VTK
        vtk_array = self.array.astype(np.float32)

        # Convertir el array a VTK
        vtk_data = numpy_support.numpy_to_vtk(
            vtk_array.ravel(),
            array_type=vtk.VTK_FLOAT
        )

        # Crear imagen VTK
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        # Para volume rendering simple, usar FixedPointVolumeRayCastMapper
        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        # Función de transferencia de opacidad - MEJORADA
        opacity_transfer = vtk.vtkPiecewiseFunction()
        data_min = np.min(self.array)
        data_max = np.max(self.array)

        # AJUSTES MEJORADOS DE OPACIDAD
        opacity_transfer.AddPoint(data_min, 0.0)
        opacity_transfer.AddPoint(-500, 0.0)  # Aire
        opacity_transfer.AddPoint(-200, 0.1)  # Pulmón/poco denso
        opacity_transfer.AddPoint(50, 0.3)  # Tejidos blandos
        opacity_transfer.AddPoint(200, 0.6)  # Tejidos más densos
        opacity_transfer.AddPoint(500, 0.8)  # Hueso/estructuras densas
        opacity_transfer.AddPoint(data_max, 1.0)

        # Función de transferencia de color - ESQUEMA MEJORADO
        color_transfer = vtk.vtkColorTransferFunction()

        # ESQUEMA DE COLOR PARA CT MÉDICO (más contrastes)
        color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)  # Negro para valores mínimos
        color_transfer.AddRGBPoint(-750, 0.0, 0.0, 0.3)  # Azul oscuro para aire
        color_transfer.AddRGBPoint(-200, 0.0, 0.5, 1.0)  # Azul claro para pulmón
        color_transfer.AddRGBPoint(0, 0.8, 0.8, 0.8)  # Gris para agua/tejidos medios
        color_transfer.AddRGBPoint(100, 1.0, 0.7, 0.4)  # Naranja para tejidos blandos
        color_transfer.AddRGBPoint(300, 1.0, 0.4, 0.2)  # Rojo-naranja para tejidos densos
        color_transfer.AddRGBPoint(600, 1.0, 0.8, 0.6)  # Amarillo claro para hueso
        color_transfer.AddRGBPoint(1000, 1.0, 1.0, 1.0)  # Blanco para hueso muy denso
        color_transfer.AddRGBPoint(data_max, 1.0, 1.0, 1.0)  # Blanco para valores máximos

        # Propiedades del volumen - MEJORADAS
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.4)  # Más luz ambiental
        volume_property.SetDiffuse(0.6)  # Más luz difusa
        volume_property.SetSpecular(0.2)  # Poco specular para mejor claridad

        # Crear volumen
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # Renderer con mejor iluminación
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1000, 800)  # Ventana más grande
        render_window.SetWindowName("Volume Rendering - DICOM 3D - COVID Scans")

        renderer.AddVolume(volume)
        renderer.SetBackground(0.0, 0.0, 0.0)  # Fondo negro para mejor contraste
        renderer.ResetCamera()

        # Interactor
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        print("Renderizado volumétrico listo. Cierra la ventana para continuar...")
        print("Controles: Click y arrastrar para rotar, R para reset, Q para salir")
        render_window.Render()
        render_window_interactor.Start()

    def surface_rendering_simple(self, threshold=None):
        """Renderizado de superficie simplificado"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print("Preparando renderizado de superficie...")

        # Convertir a tipo de datos adecuado para VTK
        vtk_array = self.array.astype(np.float32)

        # Convertir a VTK
        vtk_data = numpy_support.numpy_to_vtk(
            vtk_array.ravel(),
            array_type=vtk.VTK_FLOAT
        )

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        # Determinar threshold automáticamente para datos médicos
        if threshold is None:
            # Para datos CT, valores típicos de tejidos están entre -1000 (aire) y +1000 (hueso)
            threshold = np.percentile(self.array, 70)
            print(f"Usando threshold automático: {threshold:.2f}")

        # Marching cubes para extraer superficie
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_image)
        marching_cubes.SetValue(0, threshold)

        # Suavizar la superficie
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(20)
        smoother.SetRelaxationFactor(0.1)

        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()

        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.7, 0.4)  # Color piel/anaranjado
        actor.GetProperty().SetOpacity(0.9)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)

        # Renderer
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)
        render_window.SetWindowName("Surface Rendering - DICOM 3D")

        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.3)
        renderer.ResetCamera()

        # Interactor
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        print("Renderizado de superficie listo. Cierra la ventana para continuar...")
        print("Controles: Click y arrastrar para rotar, R para reset, Q para salir")
        render_window.Render()
        render_window_interactor.Start()

    def volume_rendering_alternative_colors(self, color_scheme="medical"):
        """Volume rendering con diferentes esquemas de color"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print("Preparando renderizado volumétrico con esquema de color alternativo...")

        vtk_array = self.array.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(vtk_array.ravel(), array_type=vtk.VTK_FLOAT)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        opacity_transfer = vtk.vtkPiecewiseFunction()
        data_min = np.min(self.array)
        data_max = np.max(self.array)

        # Opacidad estándar
        opacity_transfer.AddPoint(data_min, 0.0)
        opacity_transfer.AddPoint(-500, 0.0)
        opacity_transfer.AddPoint(-200, 0.2)
        opacity_transfer.AddPoint(50, 0.4)
        opacity_transfer.AddPoint(200, 0.7)
        opacity_transfer.AddPoint(500, 0.9)
        opacity_transfer.AddPoint(data_max, 1.0)

        color_transfer = vtk.vtkColorTransferFunction()

        if color_scheme == "hot":
            # Esquema caliente
            color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
            color_transfer.AddRGBPoint(-200, 0.3, 0.0, 0.0)
            color_transfer.AddRGBPoint(0, 0.8, 0.3, 0.0)
            color_transfer.AddRGBPoint(100, 1.0, 0.7, 0.0)
            color_transfer.AddRGBPoint(300, 1.0, 0.9, 0.3)
            color_transfer.AddRGBPoint(600, 1.0, 1.0, 0.8)
            color_transfer.AddRGBPoint(data_max, 1.0, 1.0, 1.0)

        elif color_scheme == "cool":
            # Esquema frío
            color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
            color_transfer.AddRGBPoint(-200, 0.0, 0.2, 0.5)
            color_transfer.AddRGBPoint(0, 0.2, 0.5, 0.8)
            color_transfer.AddRGBPoint(100, 0.4, 0.7, 1.0)
            color_transfer.AddRGBPoint(300, 0.6, 0.8, 1.0)
            color_transfer.AddRGBPoint(600, 0.8, 0.9, 1.0)
            color_transfer.AddRGBPoint(data_max, 1.0, 1.0, 1.0)

        else:  # medical (por defecto)
            # Esquema médico mejorado
            color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
            color_transfer.AddRGBPoint(-750, 0.1, 0.1, 0.4)
            color_transfer.AddRGBPoint(-200, 0.2, 0.5, 0.8)
            color_transfer.AddRGBPoint(0, 0.7, 0.7, 0.7)
            color_transfer.AddRGBPoint(100, 1.0, 0.6, 0.3)
            color_transfer.AddRGBPoint(300, 1.0, 0.4, 0.1)
            color_transfer.AddRGBPoint(600, 1.0, 0.9, 0.6)
            color_transfer.AddRGBPoint(data_max, 1.0, 1.0, 1.0)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.5)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.1)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1000, 800)
        render_window.SetWindowName(f"Volume Rendering - {color_scheme} scheme")

        renderer.AddVolume(volume)
        renderer.SetBackground(0.0, 0.0, 0.0)
        renderer.ResetCamera()

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        print(f"Renderizado con esquema '{color_scheme}' listo.")
        render_window.Render()
        render_window_interactor.Start()

    def volume_rendering_threshold_interactive(self):
        """Volume rendering con sliders interactivos para ajustar umbrales en tiempo real"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print("Preparando segmentación interactiva con sliders...")

        # Calcular umbrales iniciales automáticamente
        initial_lower = np.percentile(self.array, 30)
        initial_upper = np.percentile(self.array, 90)

        print(f"Umbrales iniciales - Inferior: {initial_lower:.1f}, Superior: {initial_upper:.1f}")

        # Crear el volumen VTK
        vtk_array = self.array.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(vtk_array.ravel(), array_type=vtk.VTK_FLOAT)

        self.vtk_image = vtk.vtkImageData()
        self.vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        self.vtk_image.SetSpacing([1.0, 1.0, 1.0])
        self.vtk_image.GetPointData().SetScalars(vtk_data)

        # Mapper
        self.volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        self.volume_mapper.SetInputData(self.vtk_image)

        # Funciones de transferencia
        self.opacity_transfer = vtk.vtkPiecewiseFunction()
        self.color_transfer = vtk.vtkColorTransferFunction()

        # Propiedades del volumen
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.color_transfer)
        self.volume_property.SetScalarOpacity(self.opacity_transfer)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetAmbient(0.5)
        self.volume_property.SetDiffuse(0.6)
        self.volume_property.SetSpecular(0.2)

        # Volumen
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)

        # Renderer y ventana
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1200, 900)
        self.render_window.SetWindowName("Segmentación Interactiva - Ajuste de Umbrales")

        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        self.renderer.ResetCamera()

        # Interactor
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

        # Crear sliders interactivos
        self._create_threshold_sliders(initial_lower, initial_upper)

        # Actualizar la transferencia inicial
        self._update_threshold_transfer(initial_lower, initial_upper)

        print("Segmentación interactiva lista!")
        print("Instrucciones:")
        print("  - Usa los sliders para ajustar los umbrales inferior y superior")
        print("  - Arrastra con el mouse para rotar la vista 3D")
        print("  - R: Reset de cámara")
        print("  - Q: Salir")

        self.render_window.Render()
        self.render_window_interactor.Start()

    def _create_threshold_sliders(self, initial_lower, initial_upper):
        """Crea los sliders interactivos para ajustar umbrales"""
        data_min = float(np.min(self.array))
        data_max = float(np.max(self.array))

        # Slider para umbral inferior
        lower_slider = vtk.vtkSliderRepresentation2D()
        lower_slider.SetMinimumValue(data_min)
        lower_slider.SetMaximumValue(data_max)
        lower_slider.SetValue(initial_lower)
        lower_slider.SetTitleText("Umbral Inferior")

        # Posición del slider inferior
        lower_slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        lower_slider.GetPoint1Coordinate().SetValue(0.02, 0.25)
        lower_slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        lower_slider.GetPoint2Coordinate().SetValue(0.25, 0.25)

        # Estilo del slider
        lower_slider.SetSliderLength(0.02)
        lower_slider.SetSliderWidth(0.03)
        lower_slider.SetEndCapLength(0.01)
        lower_slider.SetEndCapWidth(0.03)
        lower_slider.SetTitleHeight(0.025)
        lower_slider.SetLabelHeight(0.02)
        lower_slider.GetSliderProperty().SetColor(1.0, 1.0, 1.0)  # Color del slider
        lower_slider.GetTitleProperty().SetColor(1.0, 1.0, 1.0)  # Color del título
        lower_slider.GetLabelProperty().SetColor(1.0, 1.0, 1.0)  # Color de la etiqueta

        # Slider para umbral superior
        upper_slider = vtk.vtkSliderRepresentation2D()
        upper_slider.SetMinimumValue(data_min)
        upper_slider.SetMaximumValue(data_max)
        upper_slider.SetValue(initial_upper)
        upper_slider.SetTitleText("Umbral Superior")

        # Posición del slider superior
        upper_slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        upper_slider.GetPoint1Coordinate().SetValue(0.02, 0.15)
        upper_slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        upper_slider.GetPoint2Coordinate().SetValue(0.25, 0.15)

        # Estilo del slider
        upper_slider.SetSliderLength(0.02)
        upper_slider.SetSliderWidth(0.03)
        upper_slider.SetEndCapLength(0.01)
        upper_slider.SetEndCapWidth(0.03)
        upper_slider.SetTitleHeight(0.025)
        upper_slider.SetLabelHeight(0.02)
        upper_slider.GetSliderProperty().SetColor(1.0, 1.0, 1.0)  # Color del slider
        upper_slider.GetTitleProperty().SetColor(1.0, 1.0, 1.0)  # Color del título
        upper_slider.GetLabelProperty().SetColor(1.0, 1.0, 1.0)  # Color de la etiqueta

        # Widgets para los sliders
        self.lower_slider_widget = vtk.vtkSliderWidget()
        self.lower_slider_widget.SetInteractor(self.render_window_interactor)
        self.lower_slider_widget.SetRepresentation(lower_slider)
        self.lower_slider_widget.SetAnimationModeToAnimate()
        self.lower_slider_widget.EnabledOn()

        self.upper_slider_widget = vtk.vtkSliderWidget()
        self.upper_slider_widget.SetInteractor(self.render_window_interactor)
        self.upper_slider_widget.SetRepresentation(upper_slider)
        self.upper_slider_widget.SetAnimationModeToAnimate()
        self.upper_slider_widget.EnabledOn()

        # Callbacks para los sliders
        self.lower_slider_widget.AddObserver("InteractionEvent", self._lower_threshold_callback)
        self.upper_slider_widget.AddObserver("InteractionEvent", self._upper_threshold_callback)

    def _lower_threshold_callback(self, obj, event):
        """Callback para el slider del umbral inferior"""
        slider_widget = obj
        value = slider_widget.GetRepresentation().GetValue()
        upper_value = self.upper_slider_widget.GetRepresentation().GetValue()

        # Asegurar que el inferior no sea mayor que el superior
        if value > upper_value:
            value = upper_value
            slider_widget.GetRepresentation().SetValue(value)

        self._update_threshold_transfer(value, upper_value)
        self.render_window.Render()

    def _upper_threshold_callback(self, obj, event):
        """Callback para el slider del umbral superior"""
        slider_widget = obj
        value = slider_widget.GetRepresentation().GetValue()
        lower_value = self.lower_slider_widget.GetRepresentation().GetValue()

        # Asegurar que el superior no sea menor que el inferior
        if value < lower_value:
            value = lower_value
            slider_widget.GetRepresentation().SetValue(value)

        self._update_threshold_transfer(lower_value, value)
        self.render_window.Render()

    def _update_threshold_transfer(self, lower_threshold, upper_threshold):
        """Actualiza las funciones de transferencia basado en los umbrales"""
        data_min = float(np.min(self.array))
        data_max = float(np.max(self.array))

        # Limpiar funciones anteriores
        self.opacity_transfer.RemoveAllPoints()
        self.color_transfer.RemoveAllPoints()

        # Actualizar opacidad - hacer transparente todo fuera del rango
        self.opacity_transfer.AddPoint(data_min, 0.0)
        self.opacity_transfer.AddPoint(lower_threshold - 1, 0.0)
        self.opacity_transfer.AddPoint(lower_threshold, 0.7)

        # Punto medio para transición suave
        mid_point = (lower_threshold + upper_threshold) / 2
        self.opacity_transfer.AddPoint(mid_point, 0.9)

        self.opacity_transfer.AddPoint(upper_threshold, 0.7)
        self.opacity_transfer.AddPoint(upper_threshold + 1, 0.0)
        self.opacity_transfer.AddPoint(data_max, 0.0)

        # Actualizar color - gradiente a través del rango
        self.color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)  # Transparente
        self.color_transfer.AddRGBPoint(lower_threshold - 1, 0.0, 0.0, 0.0)  # Transparente

        # Gradiente de color azul → cian → verde → amarillo → rojo
        self.color_transfer.AddRGBPoint(lower_threshold, 0.0, 0.0, 1.0)  # Azul
        self.color_transfer.AddRGBPoint(lower_threshold + (upper_threshold - lower_threshold) * 0.25,
                                        0.0, 0.8, 1.0)  # Azul claro/Cian
        self.color_transfer.AddRGBPoint(mid_point,
                                        0.0, 1.0, 0.0)  # Verde
        self.color_transfer.AddRGBPoint(lower_threshold + (upper_threshold - lower_threshold) * 0.75,
                                        1.0, 1.0, 0.0)  # Amarillo
        self.color_transfer.AddRGBPoint(upper_threshold,
                                        1.0, 0.0, 0.0)  # Rojo

        self.color_transfer.AddRGBPoint(upper_threshold + 1, 0.0, 0.0, 0.0)  # Transparente
        self.color_transfer.AddRGBPoint(data_max, 0.0, 0.0, 0.0)  # Transparente

        # Calcular porcentaje visible
        mask = (self.array >= lower_threshold) & (self.array <= upper_threshold)
        visible_percentage = np.sum(mask) / mask.size * 100
        visible_voxels = np.sum(mask)
        total_voxels = mask.size

        # Actualizar título de la ventana con información
        self.render_window.SetWindowName(
            f"Segmentación Interactiva - Umbral: [{lower_threshold:.1f}, {upper_threshold:.1f}] - Visible: {visible_percentage:.1f}%"
        )

        # Mostrar información en consola también
        print(
            f"Umbrales: [{lower_threshold:.1f}, {upper_threshold:.1f}] | Visible: {visible_percentage:.1f}% ({visible_voxels}/{total_voxels} voxels)")





def main():
    # Ruta fija - COVID SCANS
    dicom_directory = "/home/isaac/Descargas/Covid Scans/Covid Scans/Subject (1)/98.12.2"

    print("=" * 60)
    print("VISUALIZADOR DICOM 3D - COVID SCANS")
    print("=" * 60)
    print(f"Directorio: {dicom_directory}")

    # Verificar si el directorio existe
    if not os.path.exists(dicom_directory):
        print(f"Error: El directorio '{dicom_directory}' no existe.")
        return

    # Preguntar cuántos slices usar (para empezar con pocos)
    try:
        max_slices_input = input(
            "¿Cuántos slices quieres usar? (recomendado: 10-20 para empezar, Enter para todos): ").strip()
        if max_slices_input == "":
            max_slices = None
            print("Usando todos los slices disponibles")
        else:
            max_slices = int(max_slices_input)
            print(f"Usando {max_slices} slices")
    except:
        max_slices = 15  # Valor por defecto seguro
        print(f"Usando valor por defecto: {max_slices} slices")

    # Crear visualizador
    viewer = DICOM3DViewer(dicom_directory)

    try:
        # Cargar datos
        viewer.load_dicom_series(max_slices=max_slices)

        # Mostrar preview
        print("\nMostrando preview de los slices...")
        viewer.show_slice_preview()

        # Menú de visualización
        while True:
            print("\n" + "=" * 50)
            print("OPCIONES DE VISUALIZACIÓN 3D")
            print("=" * 50)
            print("1. Volume Rendering (renderizado volumétrico)")
            print("2. Surface Rendering (renderizado de superficie)")
            print("3. Ajustar threshold para surface rendering")
            print("4. Mostrar preview de slices nuevamente")
            print("5. Información de los datos")
            print("6. Esquema de colores")
            print("7. Salir")

            choice = input("\nSelecciona una opción (1-6): ").strip()

            if choice == '1':
                print("\n" + "=" * 50)
                print("SEGMENTACIÓN INTERACTIVA POR UMBRALES")
                print("=" * 50)
                print("Iniciando visualización con sliders interactivos...")
                print("Se abrirá una ventana con dos sliders para ajustar:")
                print("  - Umbral Inferior (azul → verde)")
                print("  - Umbral Superior (verde → rojo)")
                print("\nLos valores fuera del rango seleccionado serán transparentes.")

                viewer.volume_rendering_threshold_interactive()
            elif choice == '2':
                print("Iniciando surface rendering... (esto puede tomar unos segundos)")
                viewer.surface_rendering_simple()
            elif choice == '3':
                try:
                    current_threshold = np.percentile(viewer.array, 70)
                    print(f"Threshold actual: {current_threshold:.2f}")
                    print(f"Rango de datos: {np.min(viewer.array):.2f} a {np.max(viewer.array):.2f}")
                    threshold = float(input("Ingresa el nuevo valor de threshold: "))
                    viewer.surface_rendering_simple(threshold=threshold)
                except ValueError:
                    print("Threshold no válido. Usando valor automático.")
                    viewer.surface_rendering_simple()
            elif choice == '4':
                viewer.show_slice_preview()
            elif choice == '5':
                print(f"\nINFORMACIÓN DE LOS DATOS:")
                print(f"Forma del array: {viewer.array.shape}")
                print(f"Rango de valores: {np.min(viewer.array):.2f} a {np.max(viewer.array):.2f}")
                print(f"Tipo de datos: {viewer.array.dtype}")
                print(f"Número de slices: {viewer.array.shape[0]}")
                print(f"Dimensiones de cada slice: {viewer.array.shape[1]} x {viewer.array.shape[2]}")
            # En el menú principal, agrega esta opción:
            elif choice == '6':  # O el número que prefieras
                print("\nEsquemas de color disponibles:")
                print("1. Médico (por defecto)")
                print("2. Esquema caliente")
                print("3. Esquema frío")
                color_choice = input("Selecciona esquema de color (1-3): ").strip()
                if color_choice == '2':
                    viewer.volume_rendering_alternative_colors("hot")
                elif color_choice == '3':
                    viewer.volume_rendering_alternative_colors("cool")
                else:
                    viewer.volume_rendering_alternative_colors("medical")
            elif choice == '7':
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida. Intenta de nuevo.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()