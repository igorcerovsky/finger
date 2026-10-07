#!/usr/bin/env python3
"""
export_papers_to_pdf.py
=======================
Automated publication-quality PDF generator for climbing biomechanics manuscripts:
  1. paper/paper_draft.md -> paper/pdf/paper_draft.pdf
  2. paper/practical_training_guide.md -> paper/pdf/practical_training_guide.pdf

Features:
  - LaTeX mathematical typesetting via KaTeX (inline $...$ and display $$...$$).
  - High-resolution figure integration with absolute URI resolution.
  - Academic typography with CSS Paged Media (@page, running headers, page numbers).
  - Automated PDF document outline and vector font embedding via Headless Chrome.
  - Validation of output PDFs using pypdf.

Usage:
  .venv/bin/python paper/script/export_papers_to_pdf.py
"""

import os
import sys
import re
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path

try:
    import markdown
except ImportError:
    print("Error: 'markdown' package is required. Install via: .venv/bin/pip install markdown", file=sys.stderr)
    sys.exit(1)

try:
    import pypdf
except ImportError:
    print("Error: 'pypdf' package is required. Install via: .venv/bin/pip install pypdf", file=sys.stderr)
    sys.exit(1)


def find_chrome_binary():
    """Locate Google Chrome, Chromium, or Brave executable on macOS/Linux/Windows."""
    candidates = [
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        # Linux
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        # Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
        found = shutil.which(name)
        if found:
            return found

    return None


def protect_and_convert_markdown(md_content, base_dir):
    """
    Protects LaTeX math delimiters from Markdown parsing, resolves image paths,
    and returns HTML body content.
    """
    # 1. Resolve relative image paths to absolute file:// URIs
    def resolve_image(match):
        alt_text = match.group(1)
        rel_path = match.group(2)
        abs_path = os.path.normpath(os.path.join(base_dir, rel_path))
        return f"![{alt_text}](file://{abs_path})"

    md_content = re.sub(r"\!\[(.*?)\]\((.*?)\)", resolve_image, md_content)

    # 2. Protect display math $$ ... $$ and inline math $ ... $
    math_placeholders = []

    def replace_display_math(match):
        math_placeholders.append((match.group(0), True))
        return f"@@DISPLAY_MATH_TOKEN_{len(math_placeholders) - 1}@@"

    def replace_inline_math(match):
        math_placeholders.append((match.group(0), False))
        return f"@@INLINE_MATH_TOKEN_{len(math_placeholders) - 1}@@"

    # Replace display math first
    protected = re.sub(r"\$\$(.*?)\$\$", replace_display_math, md_content, flags=re.DOTALL)
    # Replace inline math (avoiding double $ or escaped $)
    protected = re.sub(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", replace_inline_math, protected)

    # 3. Convert Markdown to HTML
    html_body = markdown.markdown(
        protected,
        extensions=["extra", "tables", "toc", "fenced_code", "sane_lists"]
    )

    # 4. Restore math formulas
    for idx, (formula, is_display) in enumerate(math_placeholders):
        token = f"@@DISPLAY_MATH_TOKEN_{idx}@@" if is_display else f"@@INLINE_MATH_TOKEN_{idx}@@"
        html_body = html_body.replace(token, formula)

    return html_body


def build_full_html(body_html, running_title, running_author="Cerovsky", doc_date="October 7, 2026", is_manuscript=True):
    """Wraps body HTML in a publication-grade HTML template with KaTeX and CSS Paged Media."""
    base_font = "'Times New Roman', Times, 'Liberation Serif', Georgia, serif" if is_manuscript else "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    header_font = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    font_size = "10pt" if is_manuscript else "9.5pt"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{running_title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<script>
window.addEventListener("DOMContentLoaded", () => {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$", right: "$", display: false}}
        ],
        throwOnError: false
    }});
}});
</script>
<style>
@page {{
    size: A4 portrait;
    margin: 24mm 18mm 20mm 18mm;
    @top-left {{
        content: "{running_title}";
        font-family: {header_font};
        font-size: 7.5pt;
        color: #64748b;
        border-bottom: 0.5pt solid #cbd5e1;
        padding-bottom: 4px;
    }}
    @top-right {{
        content: "{running_author} · {doc_date}";
        font-family: {header_font};
        font-size: 7.5pt;
        font-weight: 600;
        color: #64748b;
        border-bottom: 0.5pt solid #cbd5e1;
        padding-bottom: 4px;
    }}
    @bottom-left {{
        content: "Document Date: {doc_date}";
        font-family: {header_font};
        font-size: 7.5pt;
        color: #94a3b8;
    }}
    @bottom-center {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: {header_font};
        font-size: 8pt;
        color: #64748b;
    }}
}}

@page :first {{
    @top-left {{ content: normal; border: none; }}
    @top-right {{ content: normal; border: none; }}
    @bottom-left {{
        content: "Document Date: {doc_date}";
        font-family: {header_font};
        font-size: 7.5pt;
        color: #94a3b8;
    }}
}}

::-webkit-scrollbar {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}}

* {{
    box-sizing: border-box;
    scrollbar-width: none !important;
}}

body {{
    font-family: {base_font};
    font-size: {font_size};
    line-height: 1.55;
    color: #1e293b;
    margin: 0;
    padding: 0;
}}

h1 {{
    font-family: {header_font};
    font-size: 17pt;
    line-height: 1.28;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 12px;
    break-after: avoid;
    page-break-after: avoid;
}}

h2 {{
    font-family: {header_font};
    font-size: 12.5pt;
    line-height: 1.3;
    font-weight: 600;
    color: #1e293b;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
    break-after: avoid;
    page-break-after: avoid;
}}

h3 {{
    font-family: {header_font};
    font-size: 10.5pt;
    line-height: 1.35;
    font-weight: 600;
    color: {'#334155' if is_manuscript else '#0369a1'};
    margin-top: 16px;
    margin-bottom: 8px;
    break-after: avoid;
    page-break-after: avoid;
}}

h4 {{
    font-family: {header_font};
    font-size: 9.5pt;
    font-weight: 600;
    color: #475569;
    margin-top: 12px;
    margin-bottom: 6px;
    break-after: avoid;
    page-break-after: avoid;
}}

p, li {{
    text-align: justify;
    hyphens: auto;
    margin-top: 0;
    margin-bottom: 8px;
}}

p {{
    orphans: 3;
    widows: 3;
}}

ul, ol {{
    margin-top: 4px;
    margin-bottom: 10px;
    padding-left: 20px;
}}

li {{
    margin-bottom: 4px;
}}

hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 18px 0;
}}

blockquote {{
    margin: 12px 0;
    padding: 8px 14px;
    background-color: #f8fafc;
    border-left: 3.5px solid #0284c7;
    color: #334155;
    font-size: 9pt;
    break-inside: avoid;
    page-break-inside: avoid;
}}

blockquote p {{
    margin-bottom: 4px;
}}

blockquote p:last-child {{
    margin-bottom: 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8pt;
    font-family: {header_font};
    break-inside: avoid;
    page-break-inside: avoid;
}}

th, td {{
    padding: 4px 6px;
    border-top: 0.5pt solid #cbd5e1;
    border-bottom: 0.5pt solid #cbd5e1;
    text-align: left;
    vertical-align: middle;
}}

th {{
    font-weight: 600;
    background-color: #f1f5f9;
    border-top: 1.5pt solid #334155;
    border-bottom: 1.5pt solid #334155;
    color: #0f172a;
    font-size: 8pt;
}}

tr:hover {{
    background-color: #f8fafc;
}}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 14px auto 6px auto;
    border-radius: 2px;
    break-inside: avoid;
    page-break-inside: avoid;
}}

code {{
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 8pt;
    background-color: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
}}

pre {{
    background-color: #f8fafc;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 7.5pt;
    overflow: hidden;
    white-space: pre-wrap;
    word-break: break-word;
    break-inside: avoid;
    page-break-inside: avoid;
}}

pre code {{
    background-color: transparent;
    color: inherit;
    padding: 0;
}}

.katex-display {{
    margin: 10px 0 !important;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
}}

a {{
    color: #0284c7;
    text-decoration: none;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def render_pdf(html_path, output_pdf_path, chrome_binary):
    """Executes headless Chrome to render HTML to PDF."""
    cmd = [
        chrome_binary,
        "--headless=new",
        "--virtual-time-budget=6000",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf_path}",
        html_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Chrome PDF generation failed (exit code {result.returncode}):\n{result.stderr}")


def verify_pdf(output_pdf_path):
    """Validates the generated PDF using pypdf and returns inspection metrics."""
    reader = pypdf.PdfReader(output_pdf_path)
    page_count = len(reader.pages)
    file_size_bytes = os.path.getsize(output_pdf_path)
    file_size_kb = file_size_bytes / 1024

    image_count = sum(len(page.images) for page in reader.pages)
    return {
        "page_count": page_count,
        "file_size_kb": file_size_kb,
        "image_count": image_count
    }


def export_document(src_md, dest_pdf, running_title, is_manuscript=True, chrome_binary=None):
    """Full workflow to export a single markdown document to PDF."""
    src_path = Path(src_md).resolve()
    dest_path = Path(dest_pdf).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source markdown file not found: {src_path}")

    print(f"\n[+] Processing: {src_path.name}")
    print(f"    Target output: {dest_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    body_html = protect_and_convert_markdown(md_content, base_dir=str(src_path.parent))
    date_match = re.search(r"\*\*Date:\*\*\s*(.+)", md_content)
    doc_date = date_match.group(1).strip() if date_match else "October 7, 2026"
    full_html = build_full_html(body_html, running_title=running_title, doc_date=doc_date, is_manuscript=is_manuscript)

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(full_html)
        tmp_html_path = tmp_file.name

    try:
        render_pdf(tmp_html_path, str(dest_path), chrome_binary)
        metrics = verify_pdf(str(dest_path))
        print(f"    [OK] Successfully rendered PDF ({metrics['page_count']} pages, {metrics['file_size_kb']:.1f} KB, {metrics['image_count']} embedded figures)")
        return metrics
    finally:
        if os.path.exists(tmp_html_path):
            os.remove(tmp_html_path)


def main():
    parser = argparse.ArgumentParser(description="Export Climbing Biomechanics manuscripts to publication-grade PDFs.")
    parser.add_argument("--paper-only", action="store_true", help="Export only the academic paper draft")
    parser.add_argument("--guide-only", action="store_true", help="Export only the practical training manual")
    args = parser.parse_args()

    chrome_bin = find_chrome_binary()
    if not chrome_bin:
        print("Error: Could not find Google Chrome or Chromium executable.", file=sys.stderr)
        print("Please ensure Google Chrome is installed.", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent.parent
    paper_dir = repo_root / "paper"
    pdf_dir = paper_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if not args.guide_only:
        targets.append({
            "src": paper_dir / "paper_draft.md",
            "dest": pdf_dir / "paper_draft.pdf",
            "title": "A Three-Dimensional Biomechanical Model of the Human Finger in Rock Climbing",
            "is_manuscript": True
        })
    if not args.paper_only:
        targets.append({
            "src": paper_dir / "practical_training_guide.md",
            "dest": pdf_dir / "practical_training_guide.pdf",
            "title": "Biomechanical Manual for Finger Training & Injury Prevention in Sport Climbing",
            "is_manuscript": False
        })

    print(f"=== Climbing Biomechanics PDF Export Pipeline ===")
    print(f"Python interpreter : {sys.executable}")
    print(f"Chrome binary      : {chrome_bin}")
    print(f"Output directory   : {pdf_dir}")

    total_pages = 0
    for target in targets:
        metrics = export_document(
            src_md=target["src"],
            dest_pdf=target["dest"],
            running_title=target["title"],
            is_manuscript=target["is_manuscript"],
            chrome_binary=chrome_bin
        )
        total_pages += metrics["page_count"]

    print("\n" + "=" * 50)
    print(f"Export complete! Generated {len(targets)} PDFs totaling {total_pages} pages in {pdf_dir}/")


if __name__ == "__main__":
    main()
