
import os
import re
import nibabel as nib
import numpy as np

# --- Config ---
# ⚠️ Modifie ces chemins selon ton environnement
DATASET_DIR = "/home/perfect/Documents/GitHub/projet-AI/data_filter"       # dossier d'entrée
OUTPUT_DIR  = "/home/perfect/Documents/GitHub/projet-AI/data_t2f"   # dossier de sortie (t2f only)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# On garde UNIQUEMENT la séquence la plus importante
SEQUENCE_KEEP = "t2f"                  # T2-FLAIR
MASK_NAME     = "tumorMask"            # masque de vérité terrain

# --- Utils NIfTI ---
def load_nifti(file_path):
    img = nib.load(file_path)
    return img, img.get_fdata()

def save_nifti(data, affine, out_path, dtype=None):
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, out_path)

# Normalisation min-max pour volumes d'images (pas pour les masques)
def normalize_minmax(x):
    x = x.astype(np.float32, copy=False)
    mn, mx = np.min(x), np.max(x)
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x, dtype=np.float32)

# Matching de fichier robuste (insensible à la casse), accepte .nii ou .nii.gz
def find_first_matching_file(folder, keyword):
    pat = re.compile(re.escape(keyword), flags=re.IGNORECASE)
    for fname in sorted(os.listdir(folder)):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        if pat.search(fname):
            return os.path.join(folder, fname)
    return None

def process_timepoint(patient, timepoint, tp_path):
    # 1) Chercher le volume t2f
    t2f_path = find_first_matching_file(tp_path, SEQUENCE_KEEP)
    if t2f_path is None:
        print(f"   ⚠️  {patient}/{timepoint}: t2f introuvable → ignoré")
        return False

    # 2) (Optionnel) Chercher le masque si présent
    mask_path = find_first_matching_file(tp_path, MASK_NAME)

    # --- Traiter et sauvegarder t2f ---
    img, vol = load_nifti(t2f_path)
    vol_norm = normalize_minmax(vol)
    out_dir = os.path.join(OUTPUT_DIR, patient, timepoint)
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, f"{SEQUENCE_KEEP}_processed.nii.gz")
    save_nifti(vol_norm, img.affine, out_img, dtype=np.float32)
    print(f"   ✅ t2f → {out_img}")

    # --- Sauvegarder le masque tel quel (si disponible) ---
    if mask_path is not None:
        m_img, m = load_nifti(mask_path)
        # conserver les labels (entiers) : cast en uint8 par défaut
        out_mask = os.path.join(out_dir, f"{MASK_NAME}.nii.gz")
        save_nifti(m, m_img.affine, out_mask, dtype=np.uint8)
        print(f"   ✅ mask → {out_mask}")
    else:
        print(f"   ⚠️  {patient}/{timepoint}: masque non trouvé")

    return True

def process_dataset():
    patients = [p for p in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, p))]
    for patient in patients:
        p_path = os.path.join(DATASET_DIR, patient)
        timepoints = [tp for tp in sorted(os.listdir(p_path)) if os.path.isdir(os.path.join(p_path, tp))]

        if len(timepoints) >= 3:
            selected = timepoints[:3]  # garder seulement les 3 premiers TP (cohérent avec le code d'origine)
        else:
            selected = timepoints

        if not selected:
            continue

        print(f"--- Patient: {patient} | TP sélectionnés: {selected} ---")
        kept_any = False
        for tp in selected:
            tp_path = os.path.join(p_path, tp)
            ok = process_timepoint(patient, tp, tp_path)
            kept_any = kept_any or ok

        if not kept_any:
            print(f"   ❌ Aucun t2f valide trouvé pour {patient} → rien exporté.")

if __name__ == "__main__":
    process_dataset()
