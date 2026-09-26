# configs/config.py

PFT_COLS = [f"feat_pft_{i:02d}" for i in range(30)]
GEO_COLS = ["feat_geo_slimness", "feat_geo_roundness", "feat_geo_dispersion"]
COLOR_COLS = [
    "feat_color_R_mean", "feat_color_R_std", "feat_color_R_skew",
    "feat_color_G_mean", "feat_color_G_std", "feat_color_G_skew",
    "feat_color_B_mean", "feat_color_B_std", "feat_color_B_skew",
]
VEIN_COLS = ["feat_vein_V1", "feat_vein_V2", "feat_vein_V3"]
TEX_COLS = [
    "feat_tex_R_p2",    "feat_tex_R_p4",    "feat_tex_R_p6",
    "feat_tex_G_p2",    "feat_tex_G_p4",    "feat_tex_G_p6",
    "feat_tex_B_p2",    "feat_tex_B_p4",    "feat_tex_B_p6",
    "feat_tex_gray_p2", "feat_tex_gray_p4", "feat_tex_gray_p6",
]

FEATURE_COLS = PFT_COLS + GEO_COLS + COLOR_COLS + VEIN_COLS + TEX_COLS
META_COLS = ["path", "label", "numerical_label"]
ALL_COLS = META_COLS + FEATURE_COLS

VALID_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}