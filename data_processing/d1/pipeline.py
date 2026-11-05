"""Download, align and audit D1 CNV features. Does not train or edit D0/P0 files."""
import argparse
import json
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from data_processing.p0.common import ROOT, now, read_tsv, sha256, write_json, write_tsv

from .cna import (
    accumulate_cna, parse_case_list, parse_meta, read_clinical_table, validate_cna_meta, validate_study_meta,
)
from .download import download_studies, redact, sha256_file
from .genes import map_source_row, nodes_by_hgnc
from .select import select_samples
from .spec import (
    CANCER_TYPES, COMMIT, ONCOTREE_TYPE, REPOSITORY, STUDY_BATCH, STUDY_BATCH_ID,
    cancers_for_study, feature_columns, study_id,
)
from .tables import (
    IDENTITY, EVENT_COLUMNS, REASON_COLUMNS, apply_study_counts, build_tables, check_tables,
    empty_store, fill_absent_reasons, read_d1_tsv,
)

OUT = ROOT / "data_versions" / "d1_cnv_v1"
CODE_FILES = (
    "data_processing/d1/__init__.py",
    "data_processing/d1/spec.py",
    "data_processing/d1/download.py",
    "data_processing/d1/select.py",
    "data_processing/d1/genes.py",
    "data_processing/d1/cna.py",
    "data_processing/d1/tables.py",
    "data_processing/d1/pipeline.py",
    "data_processing/d1/build_graph.py",
    "data_processing/d1/README.md",
    "tests/test_d1.py",
)
SAMPLE_FIELDS = (
    "sample_id", "patient_id", "study", "cancer_type", "sample_type", "has_cna", "selected",
    "selection_reason", "classification_basis", "oncotree_code", "tumor_tissue_site", "tumor_type",
    "barcode_sample_type_code", "clinical_sample_class",
)
ISSUE_FIELDS = (
    "issue_type", "severity", "study", "line_number", "source_symbol", "source_entrez",
    "node_id", "hgnc_id", "mapping_method", "detail",
)
CLINICAL_COLUMNS = (
    "PATIENT_ID", "SAMPLE_ID", "ONCOTREE_CODE", "TUMOR_TISSUE_SITE", "SAMPLE_TYPE",
    "TUMOR_TYPE", "CANCER_TYPE_DETAILED",
)


def log_event(step, status, details):
    stamp = now()
    text = "\n## " + stamp + " · " + step + " · " + status + "\n\n" + details.strip() + "\n"
    for path in (OUT / "EXECUTION_LOG.md", ROOT / "data_processing/d1/EXECUTION_LOG.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# D1 CNV 执行日志\n\n时间使用 Asia/Shanghai。只记录已经执行的步骤。\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
    with (OUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": stamp, "step": step, "status": status, "details": details.strip()}, ensure_ascii=False) + "\n")


def emit_tsv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_tsv(temporary, rows, fields)
    temporary.replace(path)


def protected_rows():
    layout = json.loads((ROOT / "data_versions/p0_v1/layout_manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "data_versions/p0_v1/baseline_manifest.json").read_text(encoding="utf-8"))
    rows = [("p0_artifact:" + row["path"], ROOT / "data_versions/p0_v1" / row["path"], row["sha256"]) for row in layout["artifacts"]]
    rows += [("d0_input:" + row["path"], ROOT / row["path"], row["sha256"]) for row in baseline["files"] if row["kind"] == "data"]
    return rows


def verify_protected():
    mismatches = []
    for label, path, digest in protected_rows():
        if not path.is_file() or sha256(path) != digest:
            mismatches.append(label)
    if mismatches:
        raise RuntimeError("D0 inputs or P0 sealed artifacts changed: " + ", ".join(mismatches[:30]))
    return {"checked": len(protected_rows()), "mismatches": 0}


def load_gene_rows():
    import pickle
    with (ROOT / "data/step1/gene_toids.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    source = ROOT / "data_versions/p0_v1/gene_map.tsv"
    rows = read_tsv(source)
    genes = sorted(mapping, key=mapping.get)
    if [mapping[gene] for gene in genes] != list(range(len(genes))) or len(rows) != len(genes):
        raise ValueError("Frozen gene map and gene_toids.pkl disagree")
    for index, (gene, row) in enumerate(zip(genes, rows)):
        if int(row["node_id"]) != index or row["source_id"] != gene:
            raise ValueError("gene_map.tsv is not in frozen node order at row " + str(index))
    hgnc_manifest = json.loads((ROOT / "data_versions/p0_v1/sources/hgnc/source_manifest.json").read_text(encoding="utf-8"))
    hgnc_path = ROOT / "data_versions/p0_v1/sources/hgnc/hgnc_complete_set.tsv"
    if sha256(hgnc_path) != hgnc_manifest["sha256"]:
        raise ValueError("Pinned HGNC snapshot hash mismatch")
    from data_processing.p0.audit import HgncIndex
    index = HgncIndex(read_tsv(hgnc_path))
    return rows, index, {"gene_toids_sha256": sha256(ROOT / "data/step1/gene_toids.pkl"), "gene_map_sha256": sha256(source), "hgnc_sha256": hgnc_manifest["sha256"], "hgnc_release": hgnc_manifest["release"]}


def _require_columns(header, path):
    missing = [column for column in CLINICAL_COLUMNS if column not in header]
    if missing:
        raise ValueError(path.name + " missing columns: " + ", ".join(missing))


def build_records(study, clinical_rows, cna_samples):
    cancers = cancers_for_study(study)
    coadread = cancers == ("READ", "COAD") or set(cancers) == {"COAD", "READ"}
    by_sample = {}
    for row in clinical_rows:
        sample_id = row["SAMPLE_ID"].strip()
        if sample_id in by_sample:
            raise ValueError("Duplicate clinical sample id in " + study)
        by_sample[sample_id] = row
    cna_set = set(cna_samples)
    records = []
    for sample_id in sorted(set(by_sample) | cna_set):
        row = by_sample.get(sample_id)
        records.append({
            "sample_id": sample_id,
            "patient_id": row["PATIENT_ID"].strip() if row else "",
            "study": study,
            "study_cancer": "" if coadread else cancers[0],
            "is_coadread": coadread,
            "oncotree_code": row["ONCOTREE_CODE"].strip() if row else "",
            "tumor_tissue_site": row["TUMOR_TISSUE_SITE"].strip() if row else "",
            "tumor_type": row["TUMOR_TYPE"].strip() if row else "",
            "cancer_type_detailed": row["CANCER_TYPE_DETAILED"].strip() if row else "",
            "sample_type": row["SAMPLE_TYPE"].strip() if row else "",
            "has_cna": sample_id in cna_set,
            "in_clinical": row is not None,
        })
    selected = select_samples(records)
    clinical_ids = set(by_sample)
    overlap = len(clinical_ids & cna_set) / max(len(clinical_ids), 1)
    if overlap < 0.5:
        raise ValueError(f"{study} clinical/CNA sample overlap {overlap:.3f} is too low to treat barcodes as aligned")
    return selected


def _read_header_samples(path):
    from .download import ensure_table_bytes
    with Path(path).open("rb") as handle:
        line = handle.readline()
    ensure_table_bytes(line, Path(path).name)
    header = line.decode("utf-8-sig").rstrip("\r\n").split("\t")
    if header[:2] != ["Hugo_Symbol", "Entrez_Gene_Id"]:
        raise ValueError("Unexpected CNA header in " + str(path))
    return header[2:]


def process_sources(source_root, gene_rows, index):
    grouped = nodes_by_hgnc(gene_rows)
    store = empty_store(len(gene_rows))
    samples, issues, links = [], [], []
    studies = []

    def map_row(symbol, entrez):
        return map_source_row(symbol, entrez, index, grouped)

    for study in dict.fromkeys(study_id(cancer) for cancer in CANCER_TYPES):
        folder = Path(source_root) / study
        study_meta = validate_study_meta(parse_meta((folder / "meta_study.txt").read_text(encoding="utf-8")), study)
        cna_meta = validate_cna_meta(parse_meta((folder / "meta_cna.txt").read_text(encoding="utf-8")), study)
        expected_type = ONCOTREE_TYPE[study]
        if study_meta["type_of_cancer"] != expected_type:
            raise ValueError(f"{study} type_of_cancer is {study_meta['type_of_cancer']}, expected {expected_type}")
        if "PUBLIC" not in study_meta["groups"]:
            raise ValueError(study + " is not marked PUBLIC")
        sample_meta = parse_meta((folder / "meta_clinical_sample.txt").read_text(encoding="utf-8"))
        patient_meta = parse_meta((folder / "meta_clinical_patient.txt").read_text(encoding="utf-8"))
        if sample_meta.get("cancer_study_identifier") != study or sample_meta.get("data_filename") != "data_clinical_sample.txt":
            raise ValueError(study + " sample clinical metadata does not match the discrete study files")
        if patient_meta.get("cancer_study_identifier") != study or patient_meta.get("data_filename") != "data_clinical_patient.txt":
            raise ValueError(study + " patient clinical metadata does not match the discrete study files")
        sample_header, clinical_rows = read_clinical_table(folder / "data_clinical_sample.txt")
        _require_columns(sample_header, folder / "data_clinical_sample.txt")
        patient_header, patient_rows = read_clinical_table(folder / "data_clinical_patient.txt")
        if "PATIENT_ID" not in patient_header:
            raise ValueError(study + " patient file has no PATIENT_ID")
        patient_ids = {row["PATIENT_ID"].strip() for row in patient_rows}
        case_meta, case_ids = parse_case_list(folder / "case_lists/cases_cna.txt")
        if case_meta.get("cancer_study_identifier") != study:
            raise ValueError(study + " CNA case list belongs to another study")
        cna_samples = _read_header_samples(folder / "data_cna.txt")
        records = build_records(study, clinical_rows, cna_samples)
        selected = {cancer: sorted(row["sample_id"] for row in records if row["selected"] and row["cancer_type"] == cancer)
                    for cancer in cancers_for_study(study)}
        patients_chosen = {cancer: {row["patient_id"] for row in records if row["selected"] and row["cancer_type"] == cancer}
                           for cancer in selected}
        overlap_patients = set.intersection(*patients_chosen.values()) if len(patients_chosen) > 1 else set()
        if overlap_patients:
            raise ValueError("A patient remained selected in more than one cancer: " + study)
        missing_patients = sorted({row["patient_id"] for row in records if row["selected"]} - patient_ids)
        if missing_patients:
            issues.append({
                "issue_type": "patient_record_missing", "severity": "warning", "study": study,
                "line_number": "", "source_symbol": "", "source_entrez": "", "node_id": "", "hgnc_id": "",
                "mapping_method": "", "detail": "selected_patients_without_patient_row=" + str(len(missing_patients)),
            })
        print("process " + study + " " + json.dumps({cancer: len(ids) for cancer, ids in selected.items()}), flush=True)
        result = accumulate_cna(folder / "data_cna.txt", selected, map_row, len(gene_rows), CANCER_TYPES)
        apply_study_counts(store, result["counts"])
        samples.extend(records)
        issues.extend(result["issues"])
        for link in result["links"].values():
            links.append(link)
        case_set = set(case_ids)
        matrix_set = set(cna_samples)
        studies.append({
            "study": study,
            "cancers": list(cancers_for_study(study)),
            "study_meta": study_meta,
            "cna_meta": cna_meta,
            "clinical_samples": len(clinical_rows),
            "patient_records": len(patient_ids),
            "selected_patients_missing_patient_record": len(missing_patients),
            "selected_patients": {cancer: len(identifiers) for cancer, identifiers in selected.items()},
            "exclusion_reasons": dict(Counter(row["selection_reason"] for row in records if not row["selected"])),
            "classification_basis": dict(Counter(row["classification_basis"] for row in records)),
            "oncotree_counts": dict(Counter(row["oncotree_code"] for row in records if row["in_clinical"])),
            "case_list_samples": len(case_set),
            "matrix_samples": len(matrix_set),
            "case_list_only": len(case_set - matrix_set),
            "matrix_only": len(matrix_set - case_set),
            "cna_stats": result["stats"],
            "illegal_examples": result["illegal"],
        })
        log_event("处理 " + study, "completed", json.dumps({
            "selected_patients": studies[-1]["selected_patients"],
            "exclusions": studies[-1]["exclusion_reasons"],
            "mapped_rows": result["stats"]["accepted_rows"],
            "conflicts": result["stats"]["conflicting_nodes"],
            "illegal_cells": result["stats"]["illegal_cells"],
        }, ensure_ascii=False))
    fill_absent_reasons(store, gene_rows)
    return store, samples, issues, links, studies


def _coverage(node_ids, labels, full, any_observed):
    summary = {"nodes": len(node_ids), "all_32_observed": 0, "any_observed": 0, "by_label": {}}
    grouped = defaultdict(lambda: {"nodes": 0, "all_32_observed": 0, "any_observed": 0})
    for node_id, label in zip(node_ids, labels):
        summary["all_32_observed"] += int(full[node_id])
        summary["any_observed"] += int(any_observed[node_id])
        bucket = grouped[str(label)]
        bucket["nodes"] += 1
        bucket["all_32_observed"] += int(full[node_id])
        bucket["any_observed"] += int(any_observed[node_id])
    summary["by_label"] = dict(grouped)
    return summary


def development_coverage(full, any_observed):
    folds = []
    universes = []
    for fold in range(5):
        train = read_tsv(ROOT / f"data_versions/p0_v1/splits/fold_{fold}_train.tsv")
        validation = read_tsv(ROOT / f"data_versions/p0_v1/splits/fold_{fold}_val.tsv")
        train_ids = [int(row["node_id"]) for row in train]
        val_ids = [int(row["node_id"]) for row in validation]
        universe = set(train_ids) | set(val_ids)
        if universe & set(val_ids) != set(val_ids) or len(universe) != len(train_ids) + len(val_ids):
            raise ValueError("Development split fold is internally inconsistent")
        universes.append(frozenset(universe))
        folds.append({
            "fold": fold,
            "train": _coverage(train_ids, [row["label"] for row in train], full, any_observed),
            "validation": _coverage(val_ids, [row["label"] for row in validation], full, any_observed),
        })
    if len(set(universes)) != 1:
        raise ValueError("Development folds do not share one train/validation universe")
    return {"folds": folds, "development_nodes": len(universes[0]), "test_labels_read": False}


def _missing_paragraph(report):
    reasons = report.get("missing_reason_counts", {})
    unmapped = next((count for key, count in reasons.items() if key.endswith(":no_mapped_cna_row")), 0)
    conflicts = sum(count for key, count in reasons.items() if key.endswith(":conflicting_source_rows"))
    identity = len(report.get("unresolved_identity_nodes", []))
    names = "、".join(row["source_id"] for row in report.get("unresolved_identity_nodes", []))
    coad = next((study for study in report["studies"] if study["study"].startswith("coadread")), None)
    coad_text = ""
    if coad:
        basis = "；".join(f"{key} {value}" for key, value in sorted(coad["classification_basis"].items()))
        excluded = "，".join(f"{key} {value}" for key, value in sorted(coad["exclusion_reasons"].items())) or "无"
        coad_text = f" COAD/READ 临床分配为 {basis}；未选中原因：{excluded}。"
    return (
        f"12,944 × 32 中有 {report['table_check']['missing_cells']} 个空白单元格。"
        f" 来源行没有成功映射：每个癌种有 {unmapped} 个已解析节点在来源中没有可用的一致基因行；"
        f" {identity} 个身份未解析节点（{names}）不能从来源接收 CNV。"
        f" 来源行冲突：另有 {conflicts} 个基因-癌种位置因重复行测量不一致被隔离，其中 ZFHX3 与 LDB3 在全部癌种被隔离。"
        f" 患者测量缺失：若选中患者中某基因没有有效 GISTIC，分母可为 0；本轮真实数据中这类单元格为 0 个。"
        f" 全节点 32 列都有观测的是 {report['coverage']['all_nodes']['all_32_observed']} 个。"
        f" 没有发现 -2、-1、0、1、2 以外的取值。没有把缺失写成 0。{coad_text}"
    )


def render_readme(report):
    cancers = []
    for study in report["studies"]:
        for cancer, count in study["selected_patients"].items():
            observed = report["table_check"]["genes_observed_by_cancer"].get(cancer, 0)
            top = ", ".join(f"{code} {n}" for code, n in sorted(study["oncotree_counts"].items(), key=lambda item: (-item[1], item[0]))[:6])
            cancers.append(
                f"- {cancer}（{study['study']}；OncoTree 类型 {study['study_meta']['type_of_cancer']}；"
                f"{study['study_meta']['name']}）：选中患者 {count}；该癌种有观测的节点 {observed} / {report['nodes']}。"
                f" 临床文件 OncoTree 计数：{top}。"
            )
    lines = [
        "# D1 CNV 数据准备",
        "",
        f"状态：**数据准备完成，不可直接训练**。生成时间 {report['created_at']}。",
        "",
        "本目录是 TCGA PanCancer Atlas 2018 离散 GISTIC 频率的审计结果。缺失值保持空白，没有插补，没有生成 `data_variants/cnv_v1`，没有训练。",
        "",
        "## 来源",
        "",
        f"- 仓库：{REPOSITORY}",
        "- 使用说明：固定提交的 DataHub README 说明这是供 cBioPortal 直接载入的公开暂存数据，并用 Git LFS 保存大文件；仓库提供逐研究 zip 或 git-lfs 下载方式。本任务按固定提交逐文件核对，不使用 zip，也不凭单个 PUBLIC 标记替代来源记录。",
        f"- 固定提交：`{COMMIT}`。这是 DataHub 文件快照，不是 2018 队列的生物学日期。",
        f"- 研究批次：{STUDY_BATCH}（研究 ID 后缀 `{STUDY_BATCH_ID}`，meta 中 citation 为 TCGA, Cell 2018）。",
        "- 每个研究只使用 `meta_cna.txt` 确认为 `DISCRETE` / `gistic` 的 `data_cna.txt`。连续拷贝数文件即使存在也未下载、未混入。",
        "- LFS 地址来自该提交的 `.lfsconfig`。签名下载链接没有写入清单或日志。",
        "- 基因身份使用 P0 已固定的 HGNC 快照和 `gene_map.tsv`，没有重新下载 HGNC。",
        "",
        "## 规则",
        "",
        "- 只保留临床字段与 TCGA barcode 都表明实体原发灶（编码 01）且矩阵中确有 CNA 的样本。矛盾样本排除并写入原因。",
        "- 同一患者多个合格样本在计算事件前按 `sample_id` 字典序保留一个。",
        "- COAD/READ 来自 `coadread_tcga_pan_can_atlas_2018`。ONCOTREE 为 COAD 或 READ 时按其分类；MACR 只在部位明确为 Colon 或 Rectum 时分配；字段矛盾或无法判断则排除。同一患者不会同时进入 COAD 和 READ。",
        "- 分母是选中患者中该基因 GISTIC 属于 -2、-1、0、1、2 的人数。`NA` 不进入分母，观测到的 0 进入分母。分母为 0 时频率留空，不写成 0。",
        "- Entrez 经 HGNC 映射到唯一 `resolved` 节点。符号只在无歧义时回退。冲突重复行被隔离，不取最大值或任意一行。",
        "",
        "## 本次检查器修复",
        "",
        "- 表级验收现在先核对每张表的行数、列契约、node_id 顺序和身份字段，再逐单元格检查掩码、计数、频率、缺失原因和选中患者上限。",
        "- `python -m data_processing.d1.pipeline verify` 是只读离线验证：核对固定提交、120 个来源文件、LFS OID、登记输出、处理代码和 P0 保护文件，不会更新清单。",
        "- 旧 manifest 已保存在 `audit/manifest_before_validator_fix.json`；`audit/validator_fix_relation.json` 记录本次只修复检查器、沿用原数据并由修复后的检查器重新验证。",
        "- 训练图脚本现在要求可追溯的缺失处理协议、冻结映射和 80 列 schema，并在临时目录完成验证后才发布；当前缺失未定，不会生成正式训练图。",
        "",
        "## 命令",
        "",
        "在项目根目录执行：",
        "",
        "```powershell",
        "python -m data_processing.d1.pipeline verify",
        "python -m unittest tests.test_d1",
        "python -m data_processing.d1.build_graph --features data_versions/d1_cnv_v1/cnv_features.tsv",
        "```",
        "",
        "`build_graph` 在缺失未处理或没有显式缺失协议时会拒绝写训练图。",
        "",
        "## 癌种覆盖",
        "",
        *cancers,
        "",
        "## 缺失构成",
        "",
        _missing_paragraph(report),
        "",
        "## 尚未解决",
        "",
        report["unresolved_text"],
        "",
        "## 缺失处理选项（尚未选择）",
        "",
        "1. 平台未测到、来源行缺失或患者缺测的基因保持缺失，模型侧明确支持掩码。当前 `run_ablation.py` 拒绝非有限特征。",
        "2. 单独决定“没有成功映射的来源行”是否编码为 0，并保留这份未填充表。这会改变缺失的含义；不能在未记录决定时直接做，也不能把 `no_mapped_cna_row` 直接解释为已经证明平台未测量。",
        "3. 不删除节点，不把 32 维掩码自动拼进主实验变成 112 维。",
        "4. 不做全基因均值、中位数、标准化或 PCA。若以后按折插补，只能在该折训练基因上拟合。",
        "5. 身份未解析的节点不能靠别名自动补上 CNV。",
        "",
        "## 以后的同维对照",
        "",
        "零值对照和打乱对照都要保留这份 12,944 × 32 的缺失掩码。零值只替换已观测单元格。打乱时在每个特征列内、只在该列已观测的节点之间置换，并记录随机种子；不要把缺失位置一起打乱，避免把覆盖率变化当成 CNV 信号。本轮没有生成或训练这些对照。",
        "",
        "## 与 D0 的关系",
        "",
        "原 48 维特征来自更早的预处理，文件中没有患者清单。D1 使用上述 2018 研究批次的离散 GISTIC。癌种名称有重叠，但不能据此声称患者队列相同。",
        "",
    ]
    return "\n".join(lines)


def write_outputs(store, gene_rows, samples, issues, links, studies, identity, source_info):
    features, observed, patients, events = build_tables(gene_rows, store)
    sample_rows = [{**row, "has_cna": str(bool(row["has_cna"])).lower(), "selected": str(bool(row["selected"])).lower()} for row in samples]
    reasons = []
    reason_counts = Counter()
    for index, row in enumerate(gene_rows):
        reason_row = {"node_id": str(index), "source_id": row["source_id"]}
        for column, cancer in enumerate(CANCER_TYPES):
            reason = "" if store["den"][index][column] > 0 else store["reason"][index][column]
            reason_row[cancer] = reason
            if reason:
                reason_counts[cancer + ":" + reason] += 1
        reasons.append(reason_row)
    table_check = check_tables(features, observed, patients, events, gene_rows, reasons=reasons, sample_rows=sample_rows)
    columns = feature_columns()
    schema = []
    for position, name in enumerate(columns):
        kind, cancer = name.split(":")
        schema.append({
            "position": position, "column_name": name, "modality": kind, "cancer_type": cancer,
            "unit": "fraction_of_selected_patients",
            "statistic": "patients with GISTIC +2" if kind == "CNV_AMP" else "patients with GISTIC -2",
            "missing_rule": "blank when no selected patient has a valid discrete call or the node has no consistent mapped row; never stored as zero and not imputed",
            "source": "cBioPortal DataHub TCGA PanCancer Atlas 2018 discrete GISTIC",
            "source_release": STUDY_BATCH_ID,
            "source_commit": COMMIT,
            "study": study_id(cancer),
            "datatype": "DISCRETE",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "data_versions/p0_v1/gene_map.tsv", OUT / "gene_map.tsv")
    if sha256(OUT / "gene_map.tsv") != identity["gene_map_sha256"]:
        raise ValueError("Copied gene map hash changed")
    emit_tsv(OUT / "cnv_features.tsv", features, IDENTITY + columns)
    emit_tsv(OUT / "feature_observed.tsv", observed, IDENTITY + columns)
    emit_tsv(OUT / "cnv_patient_counts.tsv", patients, IDENTITY + CANCER_TYPES)
    emit_tsv(OUT / "cnv_event_counts.tsv", events, IDENTITY + EVENT_COLUMNS)
    emit_tsv(OUT / "feature_schema.tsv", schema, list(schema[0]))
    emit_tsv(OUT / "sample_metadata.tsv", sample_rows, SAMPLE_FIELDS)
    emit_tsv(OUT / "mapping_issues.tsv", issues, ISSUE_FIELDS)
    emit_tsv(OUT / "missing_reasons.tsv", reasons, REASON_COLUMNS)
    link_rows = []
    by_node = {int(row["node_id"]): row for row in gene_rows}
    for link in sorted(links, key=lambda item: (item["study"], int(item["node_id"]))):
        gene = by_node[int(link["node_id"])]
        link_rows.append({
            "node_id": link["node_id"], "source_id": gene["source_id"], "hgnc_id": link["hgnc_id"],
            "study": link["study"], "source_symbol": link["source_symbol"], "source_entrez": link["source_entrez"],
            "mapping_method": link["mapping_method"], "n_source_rows": link["n_source_rows"],
        })
    emit_tsv(OUT / "cnv_gene_links.tsv", link_rows, ("node_id", "source_id", "hgnc_id", "study", "source_symbol", "source_entrez", "mapping_method", "n_source_rows"))
    full = []
    any_observed = []
    for mask in observed:
        flags = [mask[column] == "1" for column in columns]
        full.append(all(flags))
        any_observed.append(any(flags))
    coverage = {
        "all_nodes": {"nodes": len(full), "all_32_observed": sum(full), "any_observed": sum(any_observed)},
        "development": development_coverage(full, any_observed),
    }
    unresolved_nodes = [{
        "node_id": int(row["node_id"]), "source_id": row["source_id"], "mapping_status": row["mapping_status"],
        "ambiguity_reason": row.get("ambiguity_reason", ""),
    } for row in gene_rows if row["mapping_status"] != "resolved"]
    status = "data_prepared_not_trainable" if table_check["missing_cells"] else "finite_matrix_ready_for_explicit_graph_build"
    unresolved_text = (
        f"缺失单元格 {table_check['missing_cells']}。身份未解析节点 {len(unresolved_nodes)} 个，状态来自 P0 gene_map 的实际 mapping_status。"
        " 原 48 维矩阵没有患者 ID，无法核实与本批 CNA 患者是否完全相同。基因组版本未写在 meta_cna.txt 中，因此记录为未知，不从其他文件名推断。"
        " KIRC、KIRP、LIHC、THCA 的 meta type_of_cancer 分别是 ccrcc、prcc、hcc、thpa；研究 ID 仍是 TCGA 项目缩写。"
        " ESCA 研究显示名是 Esophageal Adenocarcinoma，本列仍按该研究中的原发 CNA 样本汇总，没有再按组织学拆开。"
        if table_check["missing_cells"] else "32 列已经全部是有限频率。"
    )
    report = {
        "created_at": now(),
        "status": status,
        "data_preparation_complete": True,
        "trainable": False if table_check["missing_cells"] else True,
        "training_graph_generated": False,
        "nodes": len(gene_rows),
        "studies": studies,
        "table_check": table_check,
        "coverage": coverage,
        "missing_reason_counts": dict(reason_counts),
        "unresolved_identity_nodes": unresolved_nodes,
        "unresolved_text": unresolved_text,
        "illegal_examples": [item for study in studies for item in study["illegal_examples"]],
        "d0_patient_overlap": "not_verifiable_from_available_48_column_matrix",
        "protected_inputs": verify_protected(),
    }
    if table_check["missing_cells"]:
        report["trainable"] = False
    write_json(OUT / "quality_report.json", report)
    (OUT / "README.md").write_text(render_readme(report), encoding="utf-8")
    write_json(OUT / "source_manifest.json", source_info)
    outputs = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or "sources" in path.parts or path.name in {"manifest.json", "events.jsonl", "EXECUTION_LOG.md"} or path.name.endswith(".tmp"):
            continue
        outputs.append({"path": path.relative_to(OUT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    audit_relation = OUT / "audit" / "validator_fix_relation.json"
    if audit_relation.is_file():
        outputs.append({"path": audit_relation.relative_to(OUT).as_posix(), "size_bytes": audit_relation.stat().st_size, "sha256": sha256(audit_relation)})
    manifest = {
        "schema_version": 1,
        "created_at": report["created_at"],
        "pipeline": "d1_cnv_v1",
        "status": status,
        "trainable": report["trainable"],
        "training_graph_generated": False,
        "study_batch": STUDY_BATCH,
        "study_batch_id": STUDY_BATCH_ID,
        "datahub_commit": COMMIT,
        "rules": {
            "primary_tumor": "clinical primary and TCGA sample-type code 01",
            "patient_dedup": "lexicographic sample_id before event counting",
            "coadread": "ONCOTREE COAD/READ; MACR only with explicit Colon or Rectum site",
            "frequency": "GISTIC +2 or -2 over valid calls in {-2,-1,0,1,2}; missing stays blank",
            "mapping": "Entrez through pinned HGNC to one resolved node; unambiguous symbol fallback; conflicting rows quarantined",
        },
        "inputs": identity,
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "outputs": outputs,
        "controls_not_built": ["zero_value_same_mask", "within_column_permutation_same_mask"],
    }
    write_json(OUT / "manifest.json", manifest)
    return report


def make_source_info(config, files, unused, downloaded_at):
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "commit_role": "Pinned DataHub file snapshot. It is not the biological date of the 2018 cohort.",
        "study_batch": STUDY_BATCH,
        "study_batch_id": STUDY_BATCH_ID,
        "study_batch_role": "TCGA PanCancer Atlas 2018 studies identified by their study IDs, citations and PMIDs in meta_study.txt.",
        "downloaded_at": downloaded_at,
        "lfs_endpoint": config["lfs_endpoint"],
        "lfs_endpoint_source": " .lfsconfig at the pinned commit",
        "lfsconfig_text": config["lfsconfig_text"],
        "gitattributes_text": config["gitattributes_text"],
        "readme_sha256": config["readme_sha256"],
        "usage": "Public cBioPortal DataHub staging files for the named TCGA studies. No LICENSE file was present at this commit. Study metadata marks the studies PUBLIC. This preparation does not claim the same patients as the historical 48 features. FASTQ/BAM and continuous CNA matrices were not downloaded.",
        "files": [{key: value for key, value in row.items() if key != "reused" or True} for row in files],
        "unused_companions": unused,
        "signed_urls_recorded": False,
    }


def download(progress=print):
    downloaded_at = now()
    source_root = OUT / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    gitignore = source_root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")
    config, files, unused = download_studies(source_root, downloaded_at, progress)
    info = make_source_info(config, files, unused, downloaded_at)
    write_json(OUT / "source_manifest.json", info)
    log_event("下载", "completed", f"提交 {COMMIT}；文件 {len(files)} 个；LFS 端点已记录且未保存签名 URL。")
    return info


def build(source_info=None):
    if source_info is None:
        source_info = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    verify_protected()
    verify_sources(OUT, source_info)
    gene_rows, index, identity = load_gene_rows()
    store, samples, issues, links, studies = process_sources(OUT / "sources", gene_rows, index)
    report = write_outputs(store, gene_rows, samples, issues, links, studies, identity, source_info)
    log_event("构建特征表", report["status"], report["unresolved_text"])
    return report


def _load_manifest_records(out):
    manifest_path = Path(out) / "manifest.json"
    source_path = Path(out) / "source_manifest.json"
    if not manifest_path.is_file() or not source_path.is_file():
        raise ValueError("D1 manifest or source_manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return manifest, source


def verify_sources(out=OUT, source=None, allow_synthetic=False):
    out = Path(out)
    if source is None:
        source = json.loads((out / "source_manifest.json").read_text(encoding="utf-8"))
    if allow_synthetic and source.get("synthetic"):
        return {"files": 0, "lfs_files": 0, "synthetic": True}
    if source.get("commit") != COMMIT:
        raise ValueError("Source manifest commit does not match the pinned DataHub commit")
    rows = source.get("files")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("Source manifest does not contain the expected 120 file records")
    problems = []
    for row in rows:
        if row.get("commit") != COMMIT:
            problems.append("commit:" + row.get("source_path", ""))
            continue
        path = out / "sources" / row["study"] / row["filename"]
        label = row["source_path"]
        if not path.is_file():
            problems.append("missing:" + label)
            continue
        if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            problems.append("hash:" + label)
            continue
        if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
            problems.append("pointer:" + label)
            continue
        if row.get("lfs"):
            if row.get("lfs_oid") != row["sha256"] or row.get("lfs_size") != row["size"]:
                problems.append("lfs:" + label)
        elif row.get("lfs_oid") is not None or row.get("lfs_size") is not None:
            problems.append("lfs_contract:" + label)
    if problems:
        raise ValueError("Source integrity failed: " + ", ".join(problems[:30]))
    return {"files": len(rows), "lfs_files": sum(1 for row in rows if row.get("lfs"))}


def _verify_code(manifest):
    registered = manifest.get("code_sha256", {})
    missing = [name for name in CODE_FILES if name not in registered]
    if missing:
        raise ValueError("Manifest does not register code files: " + ", ".join(missing))
    problems = []
    for name, digest in registered.items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            problems.append(name)
    return problems


def verify(out=OUT, expected_status="data_prepared_not_trainable", protected=True, log=False):
    """Offline read-only verification. Default writes nothing to disk."""
    out = Path(out)
    manifest, source = _load_manifest_records(out)
    if manifest.get("datahub_commit") != COMMIT or manifest.get("pipeline") != "d1_cnv_v1":
        raise ValueError("D1 manifest identity does not match this pipeline")
    if manifest.get("status") != expected_status or manifest.get("training_graph_generated") is not False:
        raise ValueError("D1 manifest status no longer says prepared but not trainable")
    gene_rows, _index, identity = load_gene_rows()
    expected_inputs = manifest.get("inputs", {})
    if expected_inputs != identity:
        raise ValueError("Registered gene/HGNC inputs no longer match the frozen P0 files")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Manifest has no output records")
    problems = []
    for row in outputs:
        path = out / row["path"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            problems.append(row["path"])
    if problems:
        raise ValueError("Registered output hash mismatch: " + ", ".join(problems))
    code_problems = _verify_code(manifest)
    if code_problems:
        raise ValueError("Registered processing code changed without an explicit manifest relation: " + ", ".join(code_problems))
    source_check = verify_sources(out, source)
    features = read_d1_tsv(out / "cnv_features.tsv")
    observed = read_d1_tsv(out / "feature_observed.tsv")
    patients = read_d1_tsv(out / "cnv_patient_counts.tsv")
    events = read_d1_tsv(out / "cnv_event_counts.tsv")
    reasons = read_d1_tsv(out / "missing_reasons.tsv")
    sample_rows = read_d1_tsv(out / "sample_metadata.tsv")
    check = check_tables(features, observed, patients, events, gene_rows, reasons=reasons, sample_rows=sample_rows)
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    if check["nodes"] != 12944 or check["missing_cells"] != report["table_check"]["missing_cells"]:
        raise ValueError("Quality report does not match the revalidated tables")
    if report.get("status") != expected_status or report.get("trainable") is not False:
        raise ValueError("Quality report status is not prepared-but-not-trainable")
    protected_result = verify_protected() if protected else {"checked": 0, "mismatches": 0, "scope": "not_checked_for_temporary_copy"}
    result = {
        "status": expected_status,
        "trainable": False,
        "missing_cells": check["missing_cells"],
        "protected": protected_result,
        "sources": source_check,
        "outputs": len(outputs),
        "code_checked": len(manifest["code_sha256"]),
        "code_changed_since_manifest": [],
        "checked_pairs": check["checked_pairs"],
    }
    if log:
        log_event("离线完整性验证", "passed", json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))
    return result


def quality():
    """Full validation with the standard execution-log side effect."""
    return verify(log=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "build", "quality", "all", "verify"))
    args = parser.parse_args()
    try:
        if args.action in {"download", "build", "all"}:
            checked = verify_protected()
            load_gene_rows()
            log_event("保护文件与冻结身份检查", "passed", f"开始前核对 {checked['checked']} 个 D0/P0 保护文件、P0 映射及 HGNC 快照。")
        if args.action == "download":
            info = download()
            verify_sources(OUT, info)
            log_event("下载后来源检查", "passed", "下载完成后核对来源清单、大小、SHA256 与 LFS OID。")
        elif args.action == "build":
            build()
        elif args.action == "quality":
            quality()
        elif args.action == "verify":
            verify(log=False)
        else:
            info = download()
            verify_sources(OUT, info)
            log_event("下载后来源检查", "passed", "下载完成后核对来源清单、大小、SHA256 与 LFS OID。")
            build(info)
            quality()
    except Exception as exc:
        if args.action != "verify":
            log_event(args.action, "failed", redact(str(exc))[:2000] + "\n\n" + redact(traceback.format_exc())[-2000:])
        raise


if __name__ == "__main__":
    main()
