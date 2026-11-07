"""Fixed D2 sources, cancer order, classification map and output contracts.

Legacy GDC Mutect2 DR-10.0 UUIDs in preprocess_data/gene_mutation/mut_data/
gdc_manifest.mutations.mutect2.2018-11-23.txt are no longer retrievable from the
GDC Files API (404). D2 therefore uses the same pinned cBioPortal DataHub commit
as D1 for TCGA PanCancer Atlas 2018 MUTATION_EXTENDED tables, with
cases_sequenced as the cohort-level callable approximation.
"""
COMMIT = "860b63a7f2d23540c94e25dd35302e133c29f48b"
REPOSITORY = "https://github.com/cBioPortal/datahub"
RAW_ROOT = f"https://raw.githubusercontent.com/cBioPortal/datahub/{COMMIT}"
STUDY_BATCH = "TCGA PanCancer Atlas 2018"
STUDY_BATCH_ID = "tcga_pan_can_atlas_2018"
VERSION_ID = "d2_mutation_v1"

CANCER_TYPES = (
    "KIRC", "BRCA", "READ", "PRAD", "STAD", "HNSC", "LUAD", "THCA",
    "BLCA", "ESCA", "LIHC", "UCEC", "COAD", "LUSC", "CESC", "KIRP",
)

EVENT_CLASSES = ("MISSENSE", "TRUNC", "SPLICE")

# Machine-readable Variant_Classification → event class.
# Classes are mutually exclusive. Unlisted / OTHER values are audited only.
CLASSIFICATION_MAP = {
    "Missense_Mutation": "MISSENSE",
    "Nonsense_Mutation": "TRUNC",
    "Frame_Shift_Del": "TRUNC",
    "Frame_Shift_Ins": "TRUNC",
    "Nonstop_Mutation": "TRUNC",
    "Splice_Site": "SPLICE",
    # Explicitly NOT mapped into the three training classes:
    "Splice_Region": "OTHER",
    "In_Frame_Del": "OTHER",
    "In_Frame_Ins": "OTHER",
    "Silent": "OTHER",
    "Translation_Start_Site": "OTHER",
    "3'UTR": "OTHER",
    "5'UTR": "OTHER",
    "3'Flank": "OTHER",
    "5'Flank": "OTHER",
    "Intron": "OTHER",
    "RNA": "OTHER",
    "IGR": "OTHER",
    "Targeted_Region": "OTHER",
    "De_novo_Start_InFrame": "OTHER",
    "De_novo_Start_OutOfFrame": "OTHER",
}

STUDY_FILES = (
    "meta_study.txt",
    "meta_mutations.txt",
    "meta_clinical_sample.txt",
    "meta_clinical_patient.txt",
    "case_lists/cases_sequenced.txt",
    "data_clinical_sample.txt",
    "data_clinical_patient.txt",
    "data_mutations.txt",
)

MAX_LFS_BYTES = 1_200_000_000  # UCEC mutation table is ~886 MB
MISSING_TOKENS = {
    "", "na", "n/a", "null", "nan", "none", ".", "-",
    "[not available]", "[not applicable]", "[unknown]", "[discrepancy]",
}

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


def cancers_for_study(study):
    return tuple(cancer for cancer in CANCER_TYPES if study_id(cancer) == study)


def feature_columns():
    columns = []
    for cancer in CANCER_TYPES:
        for event in EVENT_CLASSES:
            columns.append(f"MUT_{event}:{cancer}")
    return tuple(columns)


def map_classification(variant_classification):
    text = (variant_classification or "").strip()
    if not text:
        return "UNKNOWN"
    return CLASSIFICATION_MAP.get(text, "UNKNOWN")
