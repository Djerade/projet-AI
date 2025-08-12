import os
import nibabel as nib
import matplotlib.pyplot as plt

# Chemin vers ton dossier de résultats
RESULTS_DIR = "/home/perfect/Documents/GitHub/projet-AI/data_filter"

def load_nifti(file_path):
    """Charge un fichier NIfTI et retourne les données et les infos d'image."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img

def display_middle_slice(data, title="Image"):
    """Affiche la coupe centrale d'un volume 3D."""
    mid_slice = data.shape[2] // 2  # Coupe au milieu sur l'axe Z
    plt.imshow(data[:, :, mid_slice], cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def browse_results(results_dir):
    """Parcourt le dossier de résultats et affiche quelques images."""
    for patient in sorted(os.listdir(results_dir)):
        patient_path = os.path.join(results_dir, patient)
        if os.path.isdir(patient_path):
            print(f"--- Patient : {patient} ---")
            for timepoint in sorted(os.listdir(patient_path)):
                tp_path = os.path.join(patient_path, timepoint)
                if os.path.isdir(tp_path):
                    for file in os.listdir(tp_path):
                        if file.endswith(".nii.gz"):
                            file_path = os.path.join(tp_path, file)
                            data, _ = load_nifti(file_path)
                            print(f"Image chargée : {file_path} - Taille : {data.shape}")
                            display_middle_slice(data, title=f"{patient} - {timepoint} - {file}")
                            # 🔹 Si tu veux limiter l'affichage, décommente ci-dessous :
                            # return

# Lancer l'affichage
browse_results(RESULTS_DIR)
