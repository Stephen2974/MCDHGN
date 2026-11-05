"""Fixed D1 sources, cancer order and file contracts.

The 2018 PanCancer Atlas study batch and the pinned DataHub commit are different facts.
"""
COMMIT = "860b63a7f2d23540c94e25dd35302e133c29f48b"
REPOSITORY = "https://github.com/cBioPortal/datahub"
RAW_ROOT = f"https://raw.githubusercontent.com/cBioPortal/datahub/{COMMIT}"
STUDY_BATCH = "TCGA PanCancer Atlas 2018"
STUDY_BATCH_ID = "tcga_pan_can_atlas_2018"

CANCER_TYPES = (
    "KIRC", "BRCA", "READ", "PRAD", "STAD", "HNSC", "LUAD", "THCA",
    "BLCA", "ESCA", "LIHC", "UCEC", "COAD", "LUSC", "CESC", "KIRP",
)

STUDY_FILES = (
    "meta_study.txt",
    "meta_cna.txt",
    "meta_clinical_sample.txt",
    "meta_clinical_patient.txt",
    "case_lists/cases_cna.txt",
    "data_clinical_sample.txt",
    "data_clinical_patient.txt",
    "data_cna.txt",
)

UNUSED_COMPANIONS = ("data_linear_cna.txt", "data_cna_hg19.seg")
MAX_LFS_BYTES = 250_000_000
GISTIC_VALUES = (-2, -1, 0, 1, 2)
MISSING_TOKENS = {
    "", "na", "n/a", "null", "nan", "none", ".", "-",
    "[not available]", "[not applicable]", "[unknown]", "[discrepancy]",
}


def study_id(cancer):
    if cancer in {"COAD", "READ"}:
        return "coadread_tcga_pan_can_atlas_2018"
    return cancer.lower() + "_tcga_pan_can_atlas_2018"


def unique_studies():
    seen = []
    for cancer in CANCER_TYPES:
        study = study_id(cancer)
        if study not in seen:
            seen.append(study)
    return tuple(seen)


# OncoTree codes stored in meta_study.txt. They are not always the TCGA project abbreviation.
ONCOTREE_TYPE = {
    "kirc_tcga_pan_can_atlas_2018": "ccrcc",
    "brca_tcga_pan_can_atlas_2018": "brca",
    "coadread_tcga_pan_can_atlas_2018": "coadread",
    "prad_tcga_pan_can_atlas_2018": "prad",
    "stad_tcga_pan_can_atlas_2018": "stad",
    "hnsc_tcga_pan_can_atlas_2018": "hnsc",
    "luad_tcga_pan_can_atlas_2018": "luad",
    "thca_tcga_pan_can_atlas_2018": "thpa",
    "blca_tcga_pan_can_atlas_2018": "blca",
    "esca_tcga_pan_can_atlas_2018": "esca",
    "lihc_tcga_pan_can_atlas_2018": "hcc",
    "ucec_tcga_pan_can_atlas_2018": "ucec",
    "lusc_tcga_pan_can_atlas_2018": "lusc",
    "cesc_tcga_pan_can_atlas_2018": "cesc",
    "kirp_tcga_pan_can_atlas_2018": "prcc",
}


def cancers_for_study(study):
    return tuple(cancer for cancer in CANCER_TYPES if study_id(cancer) == study)


def feature_columns():
    columns = []
    for cancer in CANCER_TYPES:
        columns.append(f"CNV_AMP:{cancer}")
        columns.append(f"CNV_DEL:{cancer}")
    return tuple(columns)


def cancer_index(cancer):
    return CANCER_TYPES.index(cancer)
