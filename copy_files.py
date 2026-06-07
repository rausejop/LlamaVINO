import os
import shutil

def copy_files(src_dir, dest_dir):
    """
    Copia todos los archivos de la carpeta de origen a la carpeta de destino.
    
    :param src_dir: Ruta de la carpeta de origen
    :param dest_dir: Ruta de la carpeta de destino
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copiado: {src_path} -> {dest_path}")
        elif os.path.isdir(src_path):
            copy_files(src_path, dest_path)

# Rutas de las carpetas
src_directory = "ruta/a/carpeta/origen"
dest_directory = "ruta/a/carpeta/destino"

# Ejecutar la función
copy_files(src_directory, dest_directory)
