import itk
import vtk
from vtk.util import numpy_support
import numpy as np


def visualizar_dicom_itk_vtk_2d(archivo_dicom):
    """
    Visualiza un archivo DICOM en 2D usando ITK para lectura y VTK para visualización
    """
    try:
        # Leer la imagen con ITK
        print("Leyendo imagen con ITK...")
        imagen_itk = itk.imread(archivo_dicom)
        array_imagen = itk.array_from_image(imagen_itk)

        print(f"Dimensiones: {array_imagen.shape}")
        print(f"Tipo de datos: {array_imagen.dtype}")

        # Si es 3D, tomar un slice 2D
        if array_imagen.ndim == 3:
            slice_medio = array_imagen.shape[0] // 2
            imagen_2d = array_imagen[slice_medio, :, :]
            print(f"Usando slice {slice_medio} de {array_imagen.shape[0]}")
        else:
            imagen_2d = array_imagen

        # Convertir numpy array a VTK image data
        vtk_image = numpy_support.numpy_to_vtk(
            imagen_2d.ravel(),
            array_type=vtk.VTK_FLOAT
        )

        # Crear vtkImageData
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(imagen_2d.shape[1], imagen_2d.shape[0], 1)
        image_data.GetPointData().SetScalars(vtk_image)

        # Crear mapper
        mapper = vtk.vtkImageMapper()
        mapper.SetInputData(image_data)
        mapper.SetColorWindow(255)
        mapper.SetColorLevel(128)

        # Crear actor
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)

        # Crear renderer y ventana
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)
        render_window.SetWindowName("Visualización DICOM 2D - ITK + VTK")

        # Crear interactor
        render_interactor = vtk.vtkRenderWindowInteractor()
        render_interactor.SetRenderWindow(render_window)

        # Añadir actor al renderer
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)  # Fondo gris oscuro

        # Iniciar visualización
        render_window.Render()
        render_interactor.Start()

    except Exception as e:
        print(f"Error: {e}")


# Ejemplo de uso
if __name__ == "__main__":
    #archivo_dicom = r"/home/isaac/Descargas/series-000001/image-000001.dcm"
    archivo_dicom = r"/home/isaac/Descargas/Covid Scans/Covid Scans/Subject (1)/98.12.2/56364399.dcm"
    visualizar_dicom_itk_vtk_2d(archivo_dicom)