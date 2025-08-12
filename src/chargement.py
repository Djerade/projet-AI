import os
import nibabel as nib
import numpy as np

# Dossier des données d'entrée
DATASET_DIR = "/home/perfect/Documents/GitHub/projet-AI/data"

# Dossier où sauvegarder les résultats
OUTPUT_DIR = "/home/perfect/Documents/GitHub/projet-AI/data_trier"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Séquences utiles
SEQUENCES = ["t1c", "t2f", "t2w"]
MASK_NAME = "tumorMask"

def load_nifti(file_path):
    """Charge un fichier NIfTI et retourne l'image et les données."""
    img = nib.load(file_path)
    return img, img.get_fdata()

def save_nifti(data, affine, output_path):
    """Sauvegarde un tableau numpy au format NIfTI."""
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

def process_and_save(patient, timepoint, seq_name, img_data, affine):
    """
    Exemple : traitement fictif + sauvegarde.
    Ici on normalise l'image pour exemple.
    """
    # Exemple de prétraitement : normalisation min-max
    norm_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-8)

    # Chemin de sortie
    out_dir = os.path.join(OUTPUT_DIR, patient, timepoint)
    os.makedirs(out_dir, exist_ok=True)

    # Nom du fichier de sortie
    out_path = os.path.join(out_dir, f"{seq_name}_processed.nii.gz")

    # Sauvegarde
    save_nifti(norm_data, affine, out_path)
    print(f"✅ Sauvegardé : {out_path}")

def process_dataset():
    """Charge et sauvegarde les résultats pour tous les patients."""
    for patient in sorted(os.listdir(DATASET_DIR)):
        patient_path = os.path.join(DATASET_DIR, patient)
        if os.path.isdir(patient_path):
            print(f"--- Patient : {patient} ---")
            for timepoint in sorted(os.listdir(patient_path)):
                tp_path = os.path.join(patient_path, timepoint)
                if os.path.isdir(tp_path):
                    for seq in SEQUENCES + [MASK_NAME]:
                        for file in os.listdir(tp_path):
                            if seq in file:
                                img, data = load_nifti(os.path.join(tp_path, file))
                                process_and_save(patient, timepoint, seq, data, img.affine)

# Lancer le traitement complet
process_dataset()
