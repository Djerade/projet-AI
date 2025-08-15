# Projet AI – Analyse de données médicales BraTS

Ce projet vise à analyser et visualiser des volumes médicaux (IRM cérébrales) au format NIfTI, issus du challenge BraTS, à l’aide de Python et de bibliothèques scientifiques.

## 📁 Structure du projet

- `data/` : Dossier contenant les données médicales (NIfTI, JPEG, etc.)
- `data_t2f/` : Dossier contenant les échantillons des données
- `notebooks/` : Notebooks Jupyter pour l’analyse, la visualisation et le prétraitement
- `src/` : Scripts Python pour la conversion, le chargement et le traitement des images
- `requirements.txt` : Liste des dépendances Python nécessaires

## 🚀 Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/Djerade/projet-AI.git
   cd projet-AI
   ```

2. **Créer un environnement virtuel (optionnel mais recommandé)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dépendances principales

- nibabel
- numpy
- matplotlib
- pandas
- scikit-image
- nilearn
- jupyter
- SimpleITK
- scipy

## 💡 Utilisation

- Lancez Jupyter Notebook :
  ```bash
  jupyter notebook
  ```
- Ouvrez les notebooks dans le dossier `notebooks/` pour explorer, visualiser et traiter les volumes médicaux.

## ✍️ Exemple de chargement d’un volume NIfTI

```python
import nibabel as nib
img = nib.load('data/mon_volume.nii.gz')
data = img.get_fdata()
print(data.shape)
```

## 📚 Ressources

- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/data.html)
- [Documentation Nibabel](https://nipy.org/nibabel/)

---

**Auteur·e :** [DJERADE GOLBE Parfait
**Licence :** MIT
