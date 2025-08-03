import os
import numpy as np
import nibabel as nib
import cv2

def convert_jpeg_folder_to_nifti(jpeg_folder, output_nifti_path, as_grayscale=True):
    images = []

    # Trier les fichiers image
    files = sorted([
        
        f for f in os.listdir(jpeg_folder)
      
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])
    if not files:
        print(f"❌ Aucun fichier JPEG trouvé dans : {jpeg_folder}")
        return

    print("Debug")
    for filename in files:
        print(filename)
        img_path = os.path.join(jpeg_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Erreur de lecture : {filename}")
            continue
        images.append(img)

    if not images:
        print(f"❌ Aucune image lisible dans : {jpeg_folder}")
        return

    image_stack = np.stack(images, axis=-1)  # empilement en volume
    nifti_img = nib.Nifti1Image(image_stack, affine=np.eye(4))
    nib.save(nifti_img, output_nifti_path)
    print(f"✅ {output_nifti_path} sauvegardé.")

def convert_all_subfolders(base_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subfolder_path):
            output_nifti_path = os.path.join(output_folder, f"{subdir}.nii.gz")
            print(f"🔄 Conversion du dossier : {subfolder_path}")
            convert_jpeg_folder_to_nifti(subfolder_path, output_nifti_path)

# === Exemple d'utilisation ===
# 📂 Chemin vers le dossier principal contenant les sous-dossiers
dossier_principal = "/home/perfect/Bureau/Projet IA/projet/data/data_jpeg/Testing/glioma"
# 📂 Dossier de sortie des fichiers NIfTI
dossier_sortie = "/home/perfect/Bureau/Projet IA/projet/data/data_nifti/Testing/glioma/fichiers_nifti"

convert_all_subfolders(dossier_principal, dossier_sortie)
