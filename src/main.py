# streamlit_app_tp1_tp2_to_mask_jpg_FIX.py
# --------------------------------------------------------------------
# Streamlit : l’utilisateur charge TP1 & TP2 (.nii/.nii.gz).
# Le modèle (temporal_seg_best.pt) prédit le masque de TP3.
# ➕ Option : dae_best.pt pour pré-dénosier TP1/TP2
# ➕ Sortie NIfTI (.nii.gz) ET export **JPG** (overlay ou masque seul)
# ⚠️ Fix de l’erreur "expected 5 dims, got 4" : on AJOUTE la dimension batch
# ➕ Aligneur ConvLSTM (corrige mismatch de canaux via proj 1x1 auto)
# --------------------------------------------------------------------
import os, io, tempfile
from typing import Dict

import numpy as np
import nibabel as nib
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# ==============================
# Helpers (I/O & preprocessing)
# ==============================

def pad_to_multiple(volume, multiple=8, mode='constant', value=0):
    if volume.ndim == 3:
        D, H, W = volume.shape
        pD = (0, (multiple - D % multiple) % multiple)
        pH = (0, (multiple - H % multiple) % multiple)
        pW = (0, (multiple - W % multiple) % multiple)
        return np.pad(volume, [pD, pH, pW], mode=mode, constant_values=value)
    elif volume.ndim == 4:
        C, D, H, W = volume.shape
        pD = (0, (multiple - D % multiple) % multiple)
        pH = (0, (multiple - H % multiple) % multiple)
        pW = (0, (multiple - W % multiple) % multiple)
        return np.pad(volume, [(0,0), pD, pH, pW], mode=mode, constant_values=value)
    else:
        raise ValueError("Volume must be 3D or 4D")

def pad_to_target(vol, target_shape_3d):
    if vol.ndim == 3:
        D, H, W = vol.shape
        tD, tH, tW = target_shape_3d
        return np.pad(vol, [(0, max(0, tD-D)), (0, max(0, tH-H)), (0, max(0, tW-W))], mode='constant')
    elif vol.ndim == 4:
        C, D, H, W = vol.shape
        tD, tH, tW = target_shape_3d
        return np.pad(vol, [(0,0), (0, max(0, tD-D)), (0, max(0, tH-H)), (0, max(0, tW-W))], mode='constant')
    else:
        raise ValueError("Volume must be 3D or 4D")

def align_two_volumes(vol1, vol2, multiple=8):
    v1 = pad_to_multiple(vol1, multiple=multiple)
    v2 = pad_to_multiple(vol2, multiple=multiple)
    _, D1, H1, W1 = v1.shape
    _, D2, H2, W2 = v2.shape
    tD, tH, tW = max(D1, D2), max(H1, H2), max(W1, W2)
    v1a = pad_to_target(v1, (tD, tH, tW))
    v2a = pad_to_target(v2, (tD, tH, tW))
    return v1a, v2a, (tD, tH, tW)

def load_nifti_from_upload(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.lower().endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    img = nib.load(tmp_path)
    data = img.get_fdata()
    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4,4):
        raise ValueError(f"Affine devrait être (4,4), reçu {affine.shape}")
    return data, affine, tmp_path

def zscore3d(vol):
    vol = vol.astype(np.float32, copy=False)
    m, s = (vol.mean() if vol.size else 0.0), (vol.std() + 1e-8)
    return (vol - m) / s

def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

# ----- Utils pour export JPG -----
def to_uint8(img2d):
    if img2d.size == 0:
        return np.zeros((1,1), dtype=np.uint8)
    vmin = np.percentile(img2d, 1.0)
    vmax = np.percentile(img2d, 99.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = (img2d - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

def get_slice(arr3d, plane, idx):
    D, H, W = arr3d.shape
    if plane == "axial":     # z-slice
        idx = np.clip(idx, 0, D-1)
        return arr3d[idx, :, :]
    elif plane == "coronal": # y-slice
        idx = np.clip(idx, 0, H-1)
        return arr3d[:, idx, :]
    elif plane == "sagittal":# x-slice
        idx = np.clip(idx, 0, W-1)
        return arr3d[:, :, idx]
    else:
        raise ValueError("plane must be axial/coronal/sagittal")

def make_overlay_jpg(base2d, mask2d, alpha=0.4):
    base_u8 = to_uint8(base2d)
    rgb = np.stack([base_u8]*3, axis=-1).astype(np.float32)
    red = np.zeros_like(rgb); red[..., 0] = 255.0
    m = (mask2d > 0).astype(np.float32)[..., None]
    out = (1 - alpha*m) * rgb + (alpha*m) * red
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def mask_to_jpg(mask2d):
    return Image.fromarray((mask2d > 0).astype(np.uint8) * 255)

# ==============================
# Models (DAE + Temporal Seg)
# ==============================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(base*2, base*4)
        self.out_channels = base*4
    def forward(self, x):
        x = self.enc1(x); x = self.pool1(x)
        x = self.enc2(x); x = self.pool2(x)
        z = self.bottleneck(x)
        return z

class Decoder3D(nn.Module):
    def __init__(self, out_ch=1, base=16):
        super().__init__()
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*2, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base, base)
        self.out_conv = nn.Conv3d(base, out_ch, 1)
    def forward(self, z):
        x = self.up2(z); x = self.dec2(x)
        x = self.up1(x); x = self.dec1(x)
        y = self.out_conv(x)
        return y

class DAE3D(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.encoder = Encoder3D(in_ch=in_ch, base=base)
        self.decoder = Decoder3D(out_ch=in_ch, base=base)
    def forward(self, x):
        z = self.encoder(x); rec = self.decoder(z); return rec, z

# ---- LSTM alignée pour corriger mismatch canaux ----
class ConvLSTMCell3DAligned(nn.Module):
    def __init__(self, enc_out: int, expected_in_c: int, hidden_dim: int, kernel_size=3):
        super().__init__()
        self.expected_in_c = expected_in_c
        self.hidden_dim    = hidden_dim
        self.input_dim_exp = expected_in_c - hidden_dim
        padding = kernel_size // 2
        self.proj_x = nn.Identity() if enc_out == self.input_dim_exp else nn.Conv3d(enc_out, self.input_dim_exp, 1)
        self.proj_h = nn.Identity() if enc_out == hidden_dim         else nn.Conv3d(enc_out, hidden_dim, 1)
        self.conv = nn.Conv3d(self.input_dim_exp + hidden_dim, 4*hidden_dim, kernel_size, padding=padding)
    def forward(self, x, h_prev, c_prev):
        x = self.proj_x(x)
        h_prev = self.proj_h(h_prev)
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class TemporalSegNetAligned(nn.Module):
    def __init__(self, encoder: Encoder3D, lstm_in_c_exp: int, hidden_dim: int, base=16):
        super().__init__()
        self.encoder = encoder
        enc_out = encoder.out_channels
        self.lstm = ConvLSTMCell3DAligned(enc_out, lstm_in_c_exp, hidden_dim, kernel_size=3)
        self.pre_decode = nn.Identity() if hidden_dim == base*4 else nn.Conv3d(hidden_dim, base*4, 1)
        self.decoder = Decoder3D(out_ch=1, base=base)
    def forward(self, x1, x2):
        z1 = self.encoder(x1); z2 = self.encoder(x2)
        # ⚠️ z1/z2 : [B, C, D, H, W] → on assume B=1 ici
        B, C, D, H, W = z1.shape
        h = torch.zeros(B, self.lstm.hidden_dim, D, H, W, device=z1.device, dtype=z1.dtype)
        c = torch.zeros_like(h)
        h, c = self.lstm(z1, h, c)
        h, c = self.lstm(z2, h, c)
        h = self.pre_decode(h)
        logits = self.decoder(h)
        return logits

def build_model_from_ckpt(seg_state: Dict[str, torch.Tensor], device):
    seg_state = strip_module_prefix(seg_state)
    # base depuis 1er conv de l'encodeur
    base = 16
    for k, v in seg_state.items():
        if "encoder.enc1.block.0.weight" in k and v.ndim == 5:
            base = int(v.shape[0]); break
    enc_out = base * 4
    # dims LSTM depuis conv du LSTM
    hidden_dim = enc_out
    lstm_in_c_exp = enc_out + enc_out
    for k, v in seg_state.items():
        if "lstm.conv.weight" in k and v.ndim == 5:
            out_c, in_c = v.shape[0], v.shape[1]
            hidden_dim = out_c // 4
            lstm_in_c_exp = in_c
            break
    encoder = Encoder3D(in_ch=1, base=base).to(device)
    model   = TemporalSegNetAligned(encoder=encoder,
                                    lstm_in_c_exp=lstm_in_c_exp,
                                    hidden_dim=hidden_dim,
                                    base=base).to(device)
    model.load_state_dict(seg_state, strict=False)
    return model, base, hidden_dim, lstm_in_c_exp

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="TP1+TP2 → masque TP3 (JPG export) — FIX", layout="centered")
st.title("TP1 & TP2 → prédiction masque TP3 (t2f-only) — NIfTI + JPG")

st.markdown("Chargez **TP1** et **TP2** (NIfTI), puis le checkpoint **temporal_seg_best.pt**. "
            "Optionnel : **dae_best.pt** pour pré-dénosier. "
            "Cette version corrige l’erreur `expected 5 dims, got 4` en ajoutant la **dimension batch**.")

col1, col2 = st.columns(2)
with col1:
    up_tp1 = st.file_uploader("TP1 (.nii / .nii.gz)", type=["nii", "nii.gz"], key="tp1")
with col2:
    up_tp2 = st.file_uploader("TP2 (.nii / .nii.gz)", type=["nii", "nii.gz"], key="tp2")

st.divider()
st.subheader("Modèles")
up_seg = st.file_uploader("Checkpoint segmentation (temporal_seg_best.pt/.pth)", type=["pt", "pth"])
use_dae = st.checkbox("Utiliser DAE (optionnel)", value=False)
up_dae = st.file_uploader("Checkpoint DAE (dae_best.pt/.pth)", type=["pt", "pth"]) if use_dae else None

st.divider()
st.subheader("Options d’inférence et d’export")
thr = st.slider("Seuil de binarisation", 0.1, 0.9, 0.5, 0.05)
multiple = st.selectbox("Padding multiple", options=[4, 8, 16], index=1)
affine_source = st.selectbox("Affine pour NIfTI de sortie", options=["TP2", "TP1"], index=0)

st.markdown("**Export JPG :**")
plane = st.selectbox("Plan", options=["axial", "coronal", "sagittal"], index=0)
slice_idx = st.number_input("Index de coupe (sera clampé)", min_value=0, value=0, step=1)
overlay_alpha = st.slider("Transparence overlay (rouge)", 0.1, 0.9, 0.4, 0.05)
export_mode = st.selectbox("Type de JPG", options=["Overlay (TP réf + masque)", "Masque seul"], index=0)

run = st.button("Prédire et préparer les exports")

if run:
    if up_tp1 is None or up_tp2 is None:
        st.error("Veuillez charger TP1 et TP2.")
        st.stop()
    if up_seg is None:
        st.error("Veuillez charger le checkpoint de segmentation (temporal_seg_best).")
        st.stop()
    if use_dae and up_dae is None:
        st.error("DAE activé : veuillez charger `dae_best.pt`.")
        st.stop()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"**Device** : {device}")

    # Charger volumes (bruts & affines)
    try:
        raw1, aff1, _ = load_nifti_from_upload(up_tp1)
        raw2, aff2, _ = load_nifti_from_upload(up_tp2)
    except Exception as e:
        st.error(f"Erreur chargement NIfTI : {e}")
        st.stop()

    # Normalisation + ajout canal (C=1)
    vol1 = zscore3d(raw1)[None, ...]  # (1,D,H,W)
    vol2 = zscore3d(raw2)[None, ...]  # (1,D,H,W)
    # Aligner formes (C,D,H,W)
    v1, v2, _ = align_two_volumes(vol1, vol2, multiple=multiple)

    # ⚠️ AJOUT DE LA DIMENSION BATCH → (B=1, C=1, D,H,W)
    x1_np = v1[None, ...].astype(np.float32, copy=False)  # (1,1,D,H,W)
    x2_np = v2[None, ...].astype(np.float32, copy=False)  # (1,1,D,H,W)

    # Charger modèle depuis ckpt uploadé
    try:
        seg_state = torch.load(io.BytesIO(up_seg.getvalue()), map_location=device)
    except Exception as e:
        st.error(f"Impossible de lire le checkpoint de segmentation: {e}")
        st.stop()
    model, base, hidden_dim, lstm_in_c_exp = build_model_from_ckpt(seg_state, device)
    model.eval()

    # DAE optionnel (reçoit aussi (B,C,D,H,W))
    dae = None
    if use_dae:
        try:
            dae_state = torch.load(io.BytesIO(up_dae.getvalue()), map_location=device)
            dae_state = strip_module_prefix(dae_state)
            dae_base = base
            for k, v in dae_state.items():
                if "encoder.enc1.block.0.weight" in k and v.ndim == 5:
                    dae_base = int(v.shape[0]); break
            dae = DAE3D(in_ch=1, base=dae_base).to(device)
            dae.load_state_dict(dae_state, strict=False)
            dae.eval()
            st.info(f"DAE chargé (base={dae_base}).")
        except Exception as e:
            st.error(f"Impossible de lire le checkpoint DAE: {e}")
            st.stop()

    # Vers device (B,C,D,H,W)
    x1 = torch.from_numpy(x1_np).to(device)
    x2 = torch.from_numpy(x2_np).to(device)

    # Option: débruiter (préserve shape (B,C,D,H,W))
    if dae is not None:
        with torch.no_grad():
            x1, _ = dae(x1)
            x2, _ = dae(x2)

    # Prédiction 3D
    with torch.no_grad():
        logits = model(x1, x2)                 # (1,1,D,H,W)
        probs  = torch.sigmoid(logits)[0,0].cpu().numpy()
        pred_bin = (probs > thr).astype(np.uint8)

    # Crop vers forme d’origine de TP2 (par défaut) ou TP1 si choisi
    orig_shape = raw2.shape if affine_source == "TP2" else raw1.shape
    pred_bin = pred_bin[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
    affine_out = aff2 if affine_source == "TP2" else aff1

    # ---- Export NIfTI ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_out:
        nib.save(nib.Nifti1Image(pred_bin.astype(np.uint8), affine_out), tmp_out.name)
        nii_bytes = open(tmp_out.name, "rb").read()
    st.download_button("Télécharger le masque (NIfTI)", data=nii_bytes,
                       file_name="pred_mask_tp3.nii.gz", mime="application/gzip")

    # ---- Export JPG (single slice) ----
    ref_vol = raw2 if affine_source == "TP2" else raw1
    ref_slice = get_slice(ref_vol, plane, int(slice_idx))
    mask_slice = get_slice(pred_bin, plane, int(slice_idx))

    if (ref_slice.ndim != 2) or (mask_slice.ndim != 2):
        st.warning("Slice invalide pour export JPG.")
    else:
        if export_mode.startswith("Overlay"):
            jpg_img = make_overlay_jpg(ref_slice, mask_slice, alpha=overlay_alpha)
            default_name = f"overlay_{plane}_{int(slice_idx)}.jpg"
        else:
            jpg_img = mask_to_jpg(mask_slice)
            default_name = f"mask_{plane}_{int(slice_idx)}.jpg"

        buf = io.BytesIO()
        jpg_img.save(buf, format="JPEG", quality=95, subsampling=0)
        st.image(jpg_img, caption=f"{export_mode} — {plane} slice {int(slice_idx)}", use_column_width=True)
        st.download_button("Télécharger l'image (JPG)", data=buf.getvalue(),
                           file_name=default_name, mime="image/jpeg")

    # Infos utiles
    vox_mm3 = float(abs(np.linalg.det(affine_out[:3,:3])))
    st.write(f"**Infos** : shape 3D = {pred_bin.shape} | voxel (mm³) = {vox_mm3:.3f} | "
             f"volume masque ≈ {(pred_bin.sum()*vox_mm3)/1000:.2f} mL")

# Lancer :  streamlit run streamlit_app_tp1_tp2_to_mask_jpg_FIX.py