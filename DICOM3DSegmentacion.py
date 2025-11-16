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

    def segment_by_otsu(self):
        """Segmentación usando el método de Otsu"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print("Realizando segmentación OTSU...")

        # Aplanar el array para el cálculo de Otsu
        flattened = self.array.ravel()

        # Calcular el threshold de Otsu
        from skimage.filters import threshold_otsu
        try:
            otsu_threshold = threshold_otsu(flattened)
            print(f"Threshold OTSU calculado: {otsu_threshold:.2f}")

            # Crear máscara binaria
            mask = self.array > otsu_threshold

            # Aplicar máscara
            segmented_array = np.where(mask, self.array, np.min(self.array))

            # Mostrar información
            foreground_voxels = np.sum(mask)
            total_voxels = mask.size
            percentage = (foreground_voxels / total_voxels) * 100

            print(f"Voxeles en foreground: {foreground_voxels}/{total_voxels} ({percentage:.2f}%)")

            return segmented_array, otsu_threshold, mask

        except ImportError:
            print("Error: scikit-image no está instalado. Instala con: pip install scikit-image")
            return None, None, None

    def segment_by_kmeans(self, n_clusters=3):
        """Segmentación usando K-Means clustering"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print(f"Realizando segmentación K-Means con {n_clusters} clusters...")

        try:
            from sklearn.cluster import KMeans

            # Obtener forma original
            original_shape = self.array.shape
            print(f"Forma del volumen: {original_shape}")
            print(f"Rango de intensidades: [{np.min(self.array):.2f}, {np.max(self.array):.2f}]")

            # Aplanar el array para K-Means
            flattened = self.array.ravel().reshape(-1, 1)
            print(f"Total de voxeles: {flattened.shape[0]}")

            # Si hay demasiados datos, tomar una muestra representativa
            if flattened.shape[0] > 50000:
                print("Muestreando datos para K-Means...")
                sample_indices = np.random.choice(flattened.shape[0], 50000, replace=False)
                sample_data = flattened[sample_indices]
            else:
                sample_data = flattened

            print(f"Entrenando K-Means con {sample_data.shape[0]} muestras...")

            # Aplicar K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
            kmeans.fit(sample_data)

            # Predecir para todos los voxeles
            print("Aplicando segmentación a todo el volumen...")
            labels = kmeans.predict(flattened)

            # Obtener centros de clusters
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(cluster_centers)

            print("Centros de clusters K-Means:")
            for i, idx in enumerate(sorted_indices):
                print(f"  Cluster {i}: intensidad = {cluster_centers[idx]:.2f}")

            # Usar el cluster más brillante (tejidos de interés)
            brightest_cluster = sorted_indices[-1]
            mask = (labels == brightest_cluster)
            mask_3d = mask.reshape(original_shape)

            print(
                f"Usando cluster más brillante (índice {brightest_cluster}) con intensidad {cluster_centers[brightest_cluster]:.2f}")

            # Crear array segmentado
            background_value = np.min(self.array)
            segmented_array = np.where(mask_3d, self.array, background_value)

            # Calcular estadísticas
            foreground_voxels = np.sum(mask)
            total_voxels = mask.size
            percentage = (foreground_voxels / total_voxels) * 100

            print(f"\nESTADÍSTICAS K-MEANS:")
            print(f"  - Voxeles en cluster brillante: {foreground_voxels}/{total_voxels} ({percentage:.2f}%)")
            print(f"  - Rango en segmentado: [{np.min(segmented_array):.2f}, {np.max(segmented_array):.2f}]")

            return segmented_array, kmeans, mask_3d

        except Exception as e:
            print(f"Error en segmentación K-Means: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def volume_rendering_kmeans(self, n_clusters=3):
        """Volume rendering para segmentación K-Means"""
        print(f"Iniciando segmentación K-Means con {n_clusters} clusters...")

        segmented_array, kmeans, mask = self.segment_by_kmeans(n_clusters)

        if segmented_array is None:
            print("Error: No se pudo realizar la segmentación K-Means")
            return

        print("Preparando visualización 3D para K-Means...")

        # Verificar el array segmentado
        print(f"Array segmentado - forma: {segmented_array.shape}")
        print(f"Array segmentado - rango: [{np.min(segmented_array):.2f}, {np.max(segmented_array):.2f}]")

        # Identificar valores de fondo y foreground
        background_value = np.min(segmented_array)
        foreground_mask = segmented_array > background_value
        foreground_values = segmented_array[foreground_mask]

        if len(foreground_values) == 0:
            print("ERROR: No hay valores de foreground en el array segmentado")
            return

        foreground_min = np.min(foreground_values)
        foreground_max = np.max(foreground_values)

        print(f"Foreground - rango: [{foreground_min:.2f}, {foreground_max:.2f}]")
        print(f"Foreground - voxeles: {np.sum(foreground_mask)}")

        # Convertir a VTK
        vtk_array = segmented_array.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(vtk_array.ravel(), array_type=vtk.VTK_FLOAT)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(segmented_array.shape[2], segmented_array.shape[1], segmented_array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        # Crear mapper
        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        # Función de transferencia de OPACIDAD
        opacity_transfer = vtk.vtkPiecewiseFunction()

        # Fondo completamente transparente
        opacity_transfer.AddPoint(background_value, 0.0)
        opacity_transfer.AddPoint(background_value + 0.1, 0.0)

        # Foreground con diferentes niveles de opacidad
        if foreground_min > background_value + 1:
            opacity_transfer.AddPoint(foreground_min - 0.1, 0.0)
            opacity_transfer.AddPoint(foreground_min, 0.4)
        else:
            opacity_transfer.AddPoint(background_value + 1, 0.4)

        opacity_transfer.AddPoint((foreground_min + foreground_max) / 2, 0.8)
        opacity_transfer.AddPoint(foreground_max, 1.0)

        # Función de transferencia de COLOR - Esquema específico para K-Means
        color_transfer = vtk.vtkColorTransferFunction()

        # Fondo negro/transparente
        color_transfer.AddRGBPoint(background_value, 0.0, 0.0, 0.0)
        color_transfer.AddRGBPoint(background_value + 0.1, 0.0, 0.0, 0.0)

        # Gradiente de colores cálidos para K-Means
        if foreground_min > background_value + 1:
            color_transfer.AddRGBPoint(foreground_min - 0.1, 0.0, 0.0, 0.0)
            color_transfer.AddRGBPoint(foreground_min, 1.0, 0.5, 0.0)  # Naranja
        else:
            color_transfer.AddRGBPoint(background_value + 1, 1.0, 0.5, 0.0)

        color_transfer.AddRGBPoint((foreground_min + foreground_max) * 0.4, 1.0, 0.8, 0.0)  # Amarillo-naranja
        color_transfer.AddRGBPoint((foreground_min + foreground_max) * 0.7, 1.0, 0.9, 0.4)  # Amarillo claro
        color_transfer.AddRGBPoint(foreground_max, 1.0, 1.0, 0.8)  # Amarillo muy claro

        # Propiedades del volumen
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.5)
        volume_property.SetDiffuse(0.7)
        volume_property.SetSpecular(0.3)

        # Crear volumen
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # Configurar renderer
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1000, 800)
        render_window.SetWindowName(f"Segmentación K-Means - {n_clusters} Clusters")

        renderer.AddVolume(volume)
        renderer.SetBackground(0.0, 0.1, 0.2)  # Fondo azul oscuro
        renderer.ResetCamera()

        # Interactor
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        print("=" * 60)
        print("VISUALIZACIÓN K-MEANS LISTA")
        print("=" * 60)
        print(f"Clusters: {n_clusters}")
        print(f"Voxeles visibles: {np.sum(foreground_mask)}")
        print(f"Porcentaje del volumen: {(np.sum(foreground_mask) / foreground_mask.size * 100):.2f}%")
        print("Controles: Ratón para rotar, R para reset, Q para salir")

        render_window.Render()
        render_window_interactor.Start()

    def volume_rendering_otsu(self):
        """Volume rendering para segmentación OTSU - MEJORADO"""
        segmented_array, otsu_threshold, mask = self.segment_by_otsu()

        if segmented_array is None:
            return

        print("Preparando volume rendering para segmentación OTSU...")

        # Usar el array segmentado directamente
        vtk_array = segmented_array.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(vtk_array.ravel(), array_type=vtk.VTK_FLOAT)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        # Función de transferencia para OTSU
        opacity_transfer = vtk.vtkPiecewiseFunction()
        data_min = np.min(segmented_array)
        data_max = np.max(segmented_array)

        opacity_transfer.AddPoint(data_min, 0.0)
        opacity_transfer.AddPoint(data_min + 1, 0.9)  # Opaco para segmentación
        opacity_transfer.AddPoint(data_max, 1.0)

        # Esquema de color para OTSU
        color_transfer = vtk.vtkColorTransferFunction()
        color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
        color_transfer.AddRGBPoint(data_min + 1, 0.0, 1.0, 0.0)  # Verde
        color_transfer.AddRGBPoint(data_max, 1.0, 1.0, 0.0)  # Amarillo

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.5)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.2)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1000, 800)
        render_window.SetWindowName(f"Segmentación OTSU - Threshold: {otsu_threshold:.2f}")

        renderer.AddVolume(volume)
        renderer.SetBackground(0.0, 0.0, 0.0)
        renderer.ResetCamera()

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        print(f"Segmentación OTSU lista. Threshold: {otsu_threshold:.2f}")
        render_window.Render()
        render_window_interactor.Start()

    def volume_rendering_gaussian(self, n_components=3):
        """Volume rendering para segmentación Gaussiana - CORREGIDO"""
        segmented_array, gmm, mask = self.segment_by_gaussian(n_components)

        if segmented_array is None:
            return

        print("Preparando volume rendering para segmentación Gaussiana...")

        # Usar el array segmentado directamente para el rendering
        vtk_array = segmented_array.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(vtk_array.ravel(), array_type=vtk.VTK_FLOAT)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        # Función de transferencia para segmentación gaussiana
        opacity_transfer = vtk.vtkPiecewiseFunction()
        data_min = np.min(segmented_array)
        data_max = np.max(segmented_array)
        background_value = data_min

        # Hacer transparente el fondo y opaco lo segmentado
        opacity_transfer.AddPoint(background_value, 0.0)
        opacity_transfer.AddPoint(background_value + 1, 0.9)  # Opaco para valores segmentados
        opacity_transfer.AddPoint(data_max, 1.0)

        # Esquema de color para segmentación gaussiana
        color_transfer = vtk.vtkColorTransferFunction()
        color_transfer.AddRGBPoint(background_value, 0.0, 0.0, 0.0)  # Transparente
        color_transfer.AddRGBPoint(background_value + 1, 0.2, 0.8, 0.2)  # Verde para segmentación
        color_transfer.AddRGBPoint((data_max + background_value + 1) / 2, 0.8, 0.8, 0.2)  # Amarillo
        color_transfer.AddRGBPoint(data_max, 0.8, 0.2, 0.2)  # Rojo

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.5)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.2)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1000, 800)
        render_window.SetWindowName(f"Segmentación Gaussiana - {n_components} componentes")

        renderer.AddVolume(volume)
        renderer.SetBackground(0.0, 0.0, 0.0)
        renderer.ResetCamera()

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        # Calcular estadísticas
        visible_voxels = np.sum(mask)
        total_voxels = mask.size
        percentage = (visible_voxels / total_voxels) * 100

        print(f"Segmentación Gaussiana lista.")
        print(f"Voxeles visibles: {visible_voxels}/{total_voxels} ({percentage:.2f}%)")

        render_window.Render()
        render_window_interactor.Start()

    def _volume_rendering_segmented(self, segmented_array, title):
        """Función auxiliar para volume rendering de arrays segmentados"""
        vtk_array = segmented_array.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(vtk_array.ravel(), array_type=vtk.VTK_FLOAT)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        # Función de transferencia para segmentación
        opacity_transfer = vtk.vtkPiecewiseFunction()
        data_min = np.min(segmented_array)
        data_max = np.max(segmented_array)

        # Hacer transparentes los valores de fondo
        opacity_transfer.AddPoint(data_min, 0.0)
        opacity_transfer.AddPoint(data_min + 1, 0.8)  # Opaco para valores segmentados
        opacity_transfer.AddPoint(data_max, 1.0)

        # Esquema de color para segmentación
        color_transfer = vtk.vtkColorTransferFunction()
        color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)  # Transparente
        color_transfer.AddRGBPoint(data_min + 1, 0.0, 1.0, 0.0)  # Verde para segmentación
        color_transfer.AddRGBPoint(data_max, 1.0, 1.0, 0.0)  # Amarillo para valores altos

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.5)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.2)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1000, 800)
        render_window.SetWindowName(title)

        renderer.AddVolume(volume)
        renderer.SetBackground(0.0, 0.0, 0.0)
        renderer.ResetCamera()

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        print(f"{title} lista.")
        render_window.Render()
        render_window_interactor.Start()

    def volume_rendering_multiple_schemes_interactive(self):
        """Volume rendering con múltiples esquemas de color basados en umbrales"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print("Preparando volume rendering con múltiples esquemas de color...")

        # Calcular umbrales iniciales automáticamente
        initial_lower = np.percentile(self.array, 25)
        initial_upper = np.percentile(self.array, 75)

        print(f"Umbrales iniciales - Inferior: {initial_lower:.1f}, Superior: {initial_upper:.1f}")
        print("Esquemas de color:")
        print("  - Debajo del umbral inferior: Esquema Frío (azules)")
        print("  - Entre umbrales: Esquema Médico (grises/naranjas)")
        print("  - Encima del umbral superior: Esquema Caliente (rojos/amarillos)")

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
        self.render_window.SetWindowName("Múltiples Esquemas de Color - Ajuste de Umbrales")

        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        self.renderer.ResetCamera()

        # Interactor
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

        # Crear sliders interactivos
        self._create_multischeme_sliders(initial_lower, initial_upper)

        # Actualizar la transferencia inicial
        self._update_multischeme_transfer(initial_lower, initial_upper)

        print("Visualización con múltiples esquemas lista!")
        print("Instrucciones:")
        print("  - Usa los sliders para ajustar los umbrales inferior y superior")
        print("  - Debajo del inferior: Esquema Frío (azules)")
        print("  - Entre umbrales: Esquema Médico (grises/naranjas)")
        print("  - Encima del superior: Esquema Caliente (rojos/amarillos)")
        print("  - Arrastra con el mouse para rotar la vista 3D")
        print("  - R: Reset de cámara")
        print("  - Q: Salir")

        self.render_window.Render()
        self.render_window_interactor.Start()

    def _create_multischeme_sliders(self, initial_lower, initial_upper):
        """Crea los sliders interactivos para múltiples esquemas"""
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
        lower_slider.GetSliderProperty().SetColor(0.0, 0.5, 1.0)  # Azul para esquema frío
        lower_slider.GetTitleProperty().SetColor(1.0, 1.0, 1.0)
        lower_slider.GetLabelProperty().SetColor(1.0, 1.0, 1.0)

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
        upper_slider.GetSliderProperty().SetColor(1.0, 0.5, 0.0)  # Naranja para esquema caliente
        upper_slider.GetTitleProperty().SetColor(1.0, 1.0, 1.0)
        upper_slider.GetLabelProperty().SetColor(1.0, 1.0, 1.0)

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
        self.lower_slider_widget.AddObserver("InteractionEvent", self._lower_multischeme_callback)
        self.upper_slider_widget.AddObserver("InteractionEvent", self._upper_multischeme_callback)

    def _lower_multischeme_callback(self, obj, event):
        """Callback para el slider del umbral inferior en múltiples esquemas"""
        slider_widget = obj
        value = slider_widget.GetRepresentation().GetValue()
        upper_value = self.upper_slider_widget.GetRepresentation().GetValue()

        # Asegurar que el inferior no sea mayor que el superior
        if value > upper_value:
            value = upper_value
            slider_widget.GetRepresentation().SetValue(value)

        self._update_multischeme_transfer(value, upper_value)
        self.render_window.Render()

    def _upper_multischeme_callback(self, obj, event):
        """Callback para el slider del umbral superior en múltiples esquemas"""
        slider_widget = obj
        value = slider_widget.GetRepresentation().GetValue()
        lower_value = self.lower_slider_widget.GetRepresentation().GetValue()

        # Asegurar que el superior no sea menor que el inferior
        if value < lower_value:
            value = lower_value
            slider_widget.GetRepresentation().SetValue(value)

        self._update_multischeme_transfer(lower_value, value)
        self.render_window.Render()

    def _update_multischeme_transfer(self, lower_threshold, upper_threshold):
        """Actualiza las funciones de transferencia para múltiples esquemas"""
        data_min = float(np.min(self.array))
        data_max = float(np.max(self.array))

        # Limpiar funciones anteriores
        self.opacity_transfer.RemoveAllPoints()
        self.color_transfer.RemoveAllPoints()

        # ACTUALIZAR OPACIDAD - mantener todo visible pero con variaciones
        self.opacity_transfer.AddPoint(data_min, 0.1)  # Mínima opacidad para valores bajos

        # Transición suave en el umbral inferior
        self.opacity_transfer.AddPoint(lower_threshold - (lower_threshold - data_min) * 0.2, 0.2)
        self.opacity_transfer.AddPoint(lower_threshold, 0.4)

        # Zona media - máxima opacidad
        mid_lower = lower_threshold + (upper_threshold - lower_threshold) * 0.3
        mid_upper = lower_threshold + (upper_threshold - lower_threshold) * 0.7
        self.opacity_transfer.AddPoint(mid_lower, 0.8)
        self.opacity_transfer.AddPoint(mid_upper, 0.9)

        # Transición en el umbral superior
        self.opacity_transfer.AddPoint(upper_threshold, 0.7)
        self.opacity_transfer.AddPoint(upper_threshold + (data_max - upper_threshold) * 0.2, 0.5)
        self.opacity_transfer.AddPoint(data_max, 0.3)

        # ACTUALIZAR COLOR - TRES ESQUEMAS COMBINADOS

        # 1. ESQUEMA FRÍO (debajo del umbral inferior) - Azules
        self.color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.3)  # Azul muy oscuro
        self.color_transfer.AddRGBPoint(data_min + (lower_threshold - data_min) * 0.3,
                                        0.1, 0.1, 0.5)  # Azul oscuro
        self.color_transfer.AddRGBPoint(data_min + (lower_threshold - data_min) * 0.6,
                                        0.2, 0.4, 0.8)  # Azul medio
        self.color_transfer.AddRGBPoint(lower_threshold,
                                        0.4, 0.6, 1.0)  # Azul claro

        # 2. ESQUEMA MÉDICO (entre umbrales) - Grises a Naranjas
        self.color_transfer.AddRGBPoint(lower_threshold + (upper_threshold - lower_threshold) * 0.1,
                                        0.7, 0.7, 0.7)  # Gris medio
        self.color_transfer.AddRGBPoint(lower_threshold + (upper_threshold - lower_threshold) * 0.3,
                                        0.9, 0.8, 0.6)  # Beige
        self.color_transfer.AddRGBPoint(lower_threshold + (upper_threshold - lower_threshold) * 0.5,
                                        1.0, 0.7, 0.4)  # Naranja claro
        self.color_transfer.AddRGBPoint(lower_threshold + (upper_threshold - lower_threshold) * 0.7,
                                        1.0, 0.6, 0.2)  # Naranja
        self.color_transfer.AddRGBPoint(upper_threshold,
                                        1.0, 0.5, 0.1)  # Naranja intenso

        # 3. ESQUEMA CALIENTE (encima del umbral superior) - Rojos/Amarillos
        self.color_transfer.AddRGBPoint(upper_threshold + (data_max - upper_threshold) * 0.2,
                                        1.0, 0.4, 0.0)  # Rojo-naranja
        self.color_transfer.AddRGBPoint(upper_threshold + (data_max - upper_threshold) * 0.4,
                                        1.0, 0.3, 0.0)  # Rojo
        self.color_transfer.AddRGBPoint(upper_threshold + (data_max - upper_threshold) * 0.6,
                                        1.0, 0.6, 0.2)  # Rojo-amarillo
        self.color_transfer.AddRGBPoint(upper_threshold + (data_max - upper_threshold) * 0.8,
                                        1.0, 0.8, 0.4)  # Amarillo-naranja
        self.color_transfer.AddRGBPoint(data_max,
                                        1.0, 1.0, 0.8)  # Amarillo muy claro

        # Calcular estadísticas por región
        cold_region = (self.array < lower_threshold)
        medical_region = (self.array >= lower_threshold) & (self.array <= upper_threshold)
        hot_region = (self.array > upper_threshold)

        cold_percentage = np.sum(cold_region) / cold_region.size * 100
        medical_percentage = np.sum(medical_region) / medical_region.size * 100
        hot_percentage = np.sum(hot_region) / hot_region.size * 100

        # Actualizar título de la ventana con información
        self.render_window.SetWindowName(
            f"Múltiples Esquemas - Umbrales: [{lower_threshold:.1f}, {upper_threshold:.1f}] | "
            f"Frío: {cold_percentage:.1f}% | Médico: {medical_percentage:.1f}% | Caliente: {hot_percentage:.1f}%"
        )

        print(f"Umbrales: [{lower_threshold:.1f}, {upper_threshold:.1f}] | "
              f"Frío: {cold_percentage:.1f}% | Médico: {medical_percentage:.1f}% | Caliente: {hot_percentage:.1f}%")

    def surface_rendering_double_threshold(self, lower_threshold=None, upper_threshold=None):
        """Surface rendering con umbral bajo y alto fijos"""
        if self.array is None:
            raise ValueError("Primero debe cargar la serie DICOM")

        print("Preparando surface rendering con doble umbral...")

        # Si no se proporcionan umbrales, pedirlos al usuario
        if lower_threshold is None or upper_threshold is None:
            data_min = np.min(self.array)
            data_max = np.max(self.array)
            print(f"Rango de datos: {data_min:.2f} a {data_max:.2f}")

            if lower_threshold is None:
                try:
                    lower_threshold = float(input(f"Umbral inferior (recomendado > {data_min:.2f}): "))
                except ValueError:
                    lower_threshold = np.percentile(self.array, 40)
                    print(f"Usando umbral inferior automático: {lower_threshold:.2f}")

            if upper_threshold is None:
                try:
                    upper_threshold = float(input(f"Umbral superior (recomendado < {data_max:.2f}): "))
                except ValueError:
                    upper_threshold = np.percentile(self.array, 80)
                    print(f"Usando umbral superior automático: {upper_threshold:.2f}")

        # Validar umbrales
        if lower_threshold >= upper_threshold:
            print("Error: El umbral inferior debe ser menor al superior. Usando valores automáticos.")
            lower_threshold = np.percentile(self.array, 40)
            upper_threshold = np.percentile(self.array, 80)

        print(f"Usando umbrales - Inferior: {lower_threshold:.2f}, Superior: {upper_threshold:.2f}")

        # Convertir a VTK
        vtk_array = self.array.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(vtk_array.ravel(), array_type=vtk.VTK_FLOAT)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.array.shape[2], self.array.shape[1], self.array.shape[0])
        vtk_image.SetSpacing([1.0, 1.0, 1.0])
        vtk_image.GetPointData().SetScalars(vtk_data)

        # Crear marching cubes con el rango de umbrales
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_image)
        marching_cubes.ComputeNormalsOn()

        # Usar múltiples contornos para el rango completo
        num_contours = 3  # Menos contornos para mejor rendimiento
        contour_values = np.linspace(lower_threshold, upper_threshold, num_contours)

        for i, value in enumerate(contour_values):
            marching_cubes.SetValue(i, value)

        # Suavizar la superficie (menos iteraciones para mejor rendimiento)
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(10)
        smoother.SetRelaxationFactor(0.1)

        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()

        # Actor con color basado en el valor medio del rango
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Color basado en la posición media del rango
        range_mid = (lower_threshold + upper_threshold) / 2
        data_min = np.min(self.array)
        data_max = np.max(self.array)

        # Normalizar posición media para color
        if data_max > data_min:
            color_pos = (range_mid - data_min) / (data_max - data_min)
        else:
            color_pos = 0.5

        # Esquema de color: Azul (bajo) -> Verde (medio) -> Rojo (alto)
        if color_pos < 0.33:
            # Azul a Verde
            r = 0.0
            g = color_pos * 3
            b = 1.0 - color_pos * 3
        elif color_pos < 0.66:
            # Verde a Amarillo
            r = (color_pos - 0.33) * 3
            g = 1.0
            b = 0.0
        else:
            # Amarillo a Rojo
            r = 1.0
            g = 1.0 - (color_pos - 0.66) * 3
            b = 0.0

        actor.GetProperty().SetColor(r, g, b)
        actor.GetProperty().SetOpacity(0.95)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetAmbient(0.3)

        # Renderer
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1000, 800)
        render_window.SetWindowName(f"Surface Rendering - Umbrales [{lower_threshold:.1f}, {upper_threshold:.1f}]")

        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.3)
        renderer.ResetCamera()

        # Interactor
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        # Calcular estadísticas
        mask = (self.array >= lower_threshold) & (self.array <= upper_threshold)
        visible_voxels = np.sum(mask)
        total_voxels = mask.size
        percentage = (visible_voxels / total_voxels) * 100

        print(f"Surface rendering con doble umbral listo.")
        print(f"Umbrales: [{lower_threshold:.1f}, {upper_threshold:.1f}]")
        print(f"Voxeles en el rango: {visible_voxels}/{total_voxels} ({percentage:.2f}%)")
        print(f"Color: RGB({r:.2f}, {g:.2f}, {b:.2f}) para rango medio {range_mid:.1f}")
        print("Controles: Click y arrastrar para rotar, R para reset, Q para salir")

        render_window.Render()
        render_window_interactor.Start()


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

            choice = input("\nSelecciona una opción (1-7): ").strip()

            if choice == '1':
                print("\n" + "=" * 50)
                print("MÉTODOS DE SEGMENTACIÓN")
                print("=" * 50)
                print("1. Segmentación por Umbrales Interactiva")
                print("2. Segmentación OTSU (automática)")
                print("3. Segmentación Gaussiana (GMM)")
                print("4. Múltiples Esquemas de Color por Umbrales")
                print("5. Volver al menú principal")

                seg_choice = input("Selecciona método de segmentación (1-4): ").strip()

                if seg_choice == '1':
                    print("\nIniciando segmentación interactiva por umbrales...")
                    print("Se abrirá una ventana con sliders para ajustar los umbrales.")
                    viewer.volume_rendering_threshold_interactive()

                elif seg_choice == '2':
                    print("\nIniciando segmentación OTSU...")
                    print("Calculando threshold óptimo automáticamente...")
                    viewer.volume_rendering_otsu()

                elif seg_choice == '3':
                    print("\nIniciando segmentación K-Means...")
                    try:
                        n_clusters = input("Número de clusters (Enter para 3): ").strip()
                        n_clusters = int(n_clusters) if n_clusters else 3
                        if n_clusters < 2 or n_clusters > 6:
                            print("Usando valor por defecto (3 clusters)")
                            n_clusters = 3
                    except:
                        n_clusters = 3
                        print("Usando valor por defecto (3 clusters)")

                    viewer.volume_rendering_kmeans(n_clusters)

                elif seg_choice == '4':
                    print("\nIniciando múltiples esquemas de color por umbrales...")
                    viewer.volume_rendering_multiple_schemes_interactive()

                elif seg_choice == '5':
                    continue

                else:
                    print("Opción no válida.")
            # En el menú principal, modifica la opción 2 para que sea directa:
            elif choice == '2':
                print("\n" + "=" * 50)
                print("SURFACE RENDERING CON DOBLE UMBRAL")
                print("=" * 50)
                print("Se mostrarán las superficies entre un umbral bajo y alto.")

                viewer.surface_rendering_double_threshold()
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