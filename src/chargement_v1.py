import os
import nibabel as nib
import numpy as np

# Dossier d'entrée
DATASET_DIR = "/home/perfect/Documents/GitHub/projet-AI/data"

# Dossier de sortie
OUTPUT_DIR = "/home/perfect/Documents/GitHub/projet-AI/data_filter"
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
    """Exemple de traitement simple : normalisation + sauvegarde."""
    norm_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-8)

    out_dir = os.path.join(OUTPUT_DIR, patient, timepoint)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{seq_name}_processed.nii.gz")
    save_nifti(norm_data, affine, out_path)
    print(f"✅ Sauvegardé : {out_path}")

def process_dataset():
    """Traite uniquement les patients avec >= 3 timepoints, en prenant les 3 premiers."""
    for patient in sorted(os.listdir(DATASET_DIR)):
        patient_path = os.path.join(DATASET_DIR, patient)
        if os.path.isdir(patient_path):
            timepoints = sorted([tp for tp in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, tp))])

            if len(timepoints) >= 3:
                selected_timepoints = timepoints[:3]  # Prendre seulement les 3 premiers
                print(f"--- Patient : {patient} ({len(timepoints)} TP, sélectionnés : {selected_timepoints}) ---")

                for timepoint in selected_timepoints:
                    tp_path = os.path.join(patient_path, timepoint)
                    for seq in SEQUENCES + [MASK_NAME]:
                        for file in os.listdir(tp_path):
                            if seq in file:
                                img, data = load_nifti(os.path.join(tp_path, file))
                                process_and_save(patient, timepoint, seq, data, img.affine)

# Lancer le traitement
process_dataset()
