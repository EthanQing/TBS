from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from typing import Any, Iterable
from xml.sax.saxutils import escape


DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _xml(text: Any) -> str:
    return escape(str(text if text is not None else ""), {'"': "&quot;"})


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _as_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        n = float(value)
    except Exception:
        return None
    return n if n == n else None


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def _format_metric(value: Any) -> str:
    n = _as_number(value)
    if n is None:
        return _format_value(value)
    if abs(n) >= 100:
        return f"{n:.2f}"
    return f"{n:.4f}"


def _format_size(value: Any) -> str:
    n = _as_number(value)
    return "-" if n is None else f"{n:.2f} MB"


def _format_latency(value: Any) -> str:
    n = _as_number(value)
    return "-" if n is None else f"{n:.2f} ms"


def _format_flops(value: Any) -> str:
    n = _as_number(value)
    if n is None:
        return "-"
    if n >= 1e12:
        return f"{n / 1e12:.2f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MFLOPs"
    return str(int(round(n)))


def _format_duration(seconds: Any) -> str:
    n = _as_number(seconds)
    if n is None:
        return "-"
    total = max(0, int(n))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}h {minutes}min"
    if minutes:
        return f"{minutes}min {secs}s"
    return f"{secs}s"


def _format_datetime(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "-"
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return text
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _status_label(status: Any) -> str:
    mapping = {
        "created": "等待中",
        "queued": "排队中",
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
        "deleted": "已删除",
    }
    key = _enum_value(status).strip().lower()
    return mapping.get(key, _format_value(status))


def _task_type_label(task_type: Any) -> str:
    mapping = {
        "detection": "目标检测",
        "segmentation": "图像分割",
        "classification": "图像分类",
    }
    key = _enum_value(task_type).strip().lower()
    return mapping.get(key, _format_value(task_type))


def _safe_filename(value: str) -> str:
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", str(value or "").strip())
    text = text.strip("._")
    return text or "训练报告"


def build_training_report_filename(report: dict[str, Any]) -> str:
    architecture = report.get("architecture") if isinstance(report, dict) else {}
    family = _safe_filename(str((architecture or {}).get("family") or "model"))
    variant = _safe_filename(str((architecture or {}).get("variant") or "run"))
    date = datetime.now().strftime("%Y%m%d")
    return f"训练报告_{family}_{variant}_{date}.docx"


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""


def _root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def _document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/><w:sz w:val="28"/></w:rPr>
  </w:style>
</w:styles>"""


def _run(text: Any, *, bold: bool = False, size: int | None = None, color: str | None = None) -> str:
    props = ['<w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/>']
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{int(size)}"/>')
    if color:
        props.append(f'<w:color w:val="{_xml(color)}"/>')
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{_xml(text)}</w:t></w:r>'


def _paragraph(
    text: Any = "",
    *,
    style: str | None = None,
    bold: bool = False,
    size: int | None = None,
    color: str | None = None,
) -> str:
    text = _format_value(text)
    if "\n" in text:
        return "".join(
            _paragraph(line, style=style, bold=bold, size=size, color=color)
            for line in text.splitlines()
        )
    ppr = f'<w:pPr><w:pStyle w:val="{_xml(style)}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{_run(text, bold=bold, size=size, color=color)}</w:p>"


def _heading(text: str) -> str:
    return _paragraph(text, style="Heading1")


def _cell(text: Any, *, header: bool = False, width: int | None = None) -> str:
    shading = '<w:shd w:fill="F1F5F9"/>' if header else ""
    width_xml = f'<w:tcW w:w="{int(width)}" w:type="dxa"/>' if width else ""
    tcpr = f"<w:tcPr>{width_xml}{shading}<w:vAlign w:val=\"center\"/></w:tcPr>"
    body = _paragraph(_format_value(text), bold=header)
    return f"<w:tc>{tcpr}{body}</w:tc>"


def _table(rows: Iterable[Iterable[Any]], *, header: bool = True, widths: list[int] | None = None) -> str:
    rows_list = [list(row) for row in rows]
    if not rows_list:
        rows_list = [["暂无数据", ""]]
    col_count = max(len(row) for row in rows_list)
    widths = widths or [9000 // max(col_count, 1)] * col_count
    grid = "".join(f'<w:gridCol w:w="{int(widths[min(i, len(widths) - 1)])}"/>' for i in range(col_count))
    tr_xml = []
    for idx, row in enumerate(rows_list):
        padded = row + [""] * (col_count - len(row))
        cells = "".join(
            _cell(value, header=bool(header and idx == 0), width=widths[min(i, len(widths) - 1)])
            for i, value in enumerate(padded)
        )
        tr_xml.append(f"<w:tr>{cells}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblW w:w="0" w:type="auto"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        "</w:tblBorders>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{''.join(tr_xml)}"
        "</w:tbl>"
    )


def _kv_rows(items: list[tuple[str, Any]]) -> list[list[str]]:
    return [["项目", "内容"], *[[label, _format_value(value)] for label, value in items]]


def _dict_rows(mapping: Any, *, metric: bool = False) -> list[list[str]]:
    if not isinstance(mapping, dict) or not mapping:
        return [["项目", "内容"], ["暂无数据", "-"]]
    rows = [["项目", "内容"]]
    for key in sorted(mapping.keys(), key=lambda x: str(x).lower()):
        value = mapping.get(key)
        rows.append([str(key), _format_metric(value) if metric else _format_value(value)])
    return rows


def _document_xml(report: dict[str, Any]) -> str:
    basic = report.get("basic") or {}
    dataset = report.get("dataset") or {}
    architecture = report.get("architecture") or {}
    parameters = report.get("parameters") or {}
    metrics = report.get("metrics") or {}
    artifacts = report.get("artifacts") or {}

    parts: list[str] = [
        _paragraph("训练结果报告", style="Title"),
        _paragraph(f"任务名称：{basic.get('name') or basic.get('run_id') or '-'}"),
        _paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        _heading("一、基本信息"),
        _table(
            _kv_rows(
                [
                    ("Run ID", basic.get("run_id")),
                    ("任务名称", basic.get("name")),
                    ("状态", _status_label(basic.get("status"))),
                    ("数据集", dataset.get("dataset_name")),
                    ("数据集 ID", dataset.get("dataset_id")),
                    ("数据集版本", dataset.get("dataset_version")),
                    ("框架", basic.get("framework_label")),
                    ("训练引擎", basic.get("engine")),
                    ("训练时长", _format_duration(basic.get("duration_seconds"))),
                    ("创建时间", _format_datetime(basic.get("created_at"))),
                    ("开始时间", _format_datetime(basic.get("started_at"))),
                    ("完成时间", _format_datetime(basic.get("finished_at"))),
                ]
            ),
            widths=[2600, 6600],
        ),
        _heading("二、模型架构选型"),
        _table(
            _kv_rows(
                [
                    ("架构 ID", architecture.get("architecture_id")),
                    ("家族", architecture.get("family")),
                    ("变体", architecture.get("variant")),
                    ("任务类型", _task_type_label(architecture.get("task_type"))),
                    ("训练引擎", basic.get("engine")),
                    ("预训练权重", architecture.get("pretrained_path")),
                    ("描述", architecture.get("description")),
                ]
            ),
            widths=[2600, 6600],
        ),
        _heading("三、训练参数配置"),
        _table(
            _kv_rows(
                [
                    ("epochs", parameters.get("epochs")),
                    ("batch_size", parameters.get("batch_size")),
                    ("image_size", parameters.get("image_size")),
                    ("learning_rate", parameters.get("learning_rate")),
                    ("patience", parameters.get("patience")),
                    ("device", parameters.get("device")),
                    ("workers", parameters.get("workers")),
                    ("optimizer", parameters.get("optimizer")),
                    ("use_pretrained", parameters.get("use_pretrained")),
                    ("save_period", parameters.get("save_period")),
                ]
            ),
            widths=[2600, 6600],
        ),
        _paragraph("增强配置", bold=True, size=24),
        _table(_dict_rows(parameters.get("augmentation")), widths=[3000, 6200]),
        _paragraph("损失权重", bold=True, size=24),
        _table(_dict_rows(parameters.get("loss_weights")), widths=[3000, 6200]),
        _paragraph("框架特有 / 其他参数", bold=True, size=24),
        _table(_dict_rows(parameters.get("additional_params")), widths=[3000, 6200]),
        _heading("四、模型最终指标"),
        _paragraph("核心指标", bold=True, size=24),
        _table(_dict_rows(metrics.get("core_metrics"), metric=True), widths=[3000, 6200]),
        _paragraph("Best Metrics", bold=True, size=24),
        _table(_dict_rows(metrics.get("best_metrics"), metric=True), widths=[4200, 5000]),
        _paragraph("Final Metrics", bold=True, size=24),
        _table(_dict_rows(metrics.get("final_metrics"), metric=True), widths=[4200, 5000]),
        _heading("五、模型产物"),
        _table(
            _kv_rows(
                [
                    ("最佳权重", artifacts.get("best_weights_path")),
                    ("最终权重", artifacts.get("last_weights_path")),
                    ("模型大小", _format_size(artifacts.get("model_size_mb"))),
                    ("推理耗时", _format_latency(artifacts.get("inference_time_ms"))),
                    ("FLOPs", _format_flops(artifacts.get("flops"))),
                ]
            ),
            widths=[2600, 6600],
        ),
    ]

    body = "".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:xml="http://www.w3.org/XML/1998/namespace">'
        f"<w:body>{body}"
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
        "</w:body></w:document>"
    )


def build_training_report_docx(report: dict[str, Any]) -> bytes:
    """Build an editable .docx report without requiring external dependencies."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml())
        zf.writestr("_rels/.rels", _root_rels_xml())
        zf.writestr("word/_rels/document.xml.rels", _document_rels_xml())
        zf.writestr("word/styles.xml", _styles_xml())
        zf.writestr("word/document.xml", _document_xml(report))
    return buf.getvalue()
