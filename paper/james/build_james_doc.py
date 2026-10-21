#!/usr/bin/env python3
"""
Script to build cerovsky_practical_training_guide_sk.docx
from the template 1_ICI_titulka_požiadavky_na_form_úpravu.docx
and the Slovak translation of practical_training_guide.md.
"""

import os
import docx
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

def make_toc_header():
    """Create the styled 'Obsah' heading for the Table of Contents."""
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    pStyle = OxmlElement('w:pStyle')
    pStyle.set(qn('w:val'), 'Hlavikaobsahu')
    pPr.append(pStyle)
    pageBreakBefore = OxmlElement('w:pageBreakBefore')
    pPr.append(pageBreakBefore)
    rPr_p = OxmlElement('w:rPr')
    color_p = OxmlElement('w:color')
    color_p.set(qn('w:val'), 'auto')
    rPr_p.append(color_p)
    pPr.append(rPr_p)
    p.append(pPr)
    
    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), 'auto')
    rPr.append(color)
    r.append(rPr)
    t = OxmlElement('w:t')
    t.text = 'Obsah'
    r.append(t)
    p.append(r)
    return p

def make_empty_p():
    """Create an empty spacing paragraph for TOC."""
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    rPr = OxmlElement('w:rPr')
    lang = OxmlElement('w:lang')
    lang.set(qn('w:eastAsia'), 'sk-SK')
    rPr.append(lang)
    pPr.append(rPr)
    p.append(pPr)
    return p

def make_toc_entry(title, page, level=1):
    """
    Create a styled Table of Contents entry matching the template:
    Level 1: style Obsah1, bold, dot leader, bold page number.
    Level 2: style Obsah2, left indent 216 twips, normal font, dot leader, page number.
    """
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    pStyle = OxmlElement('w:pStyle')
    pStyle.set(qn('w:val'), f'Obsah{level}')
    pPr.append(pStyle)
    if level == 1:
        ind = OxmlElement('w:ind')
        ind.set(qn('w:left'), '0')
        ind.set(qn('w:firstLine'), '0')
        pPr.append(ind)
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), 'left')
        pPr.append(jc)
        tabs = OxmlElement('w:tabs')
        tab = OxmlElement('w:tab')
        tab.set(qn('w:val'), 'right')
        tab.set(qn('w:leader'), 'dot')
        tab.set(qn('w:pos'), '9071')
        tabs.append(tab)
        pPr.append(tabs)
        rPr_p = OxmlElement('w:rPr')
        b = OxmlElement('w:b')
        bCs = OxmlElement('w:bCs')
        rPr_p.append(b)
        rPr_p.append(bCs)
        pPr.append(rPr_p)
    elif level == 2:
        ind = OxmlElement('w:ind')
        ind.set(qn('w:left'), '216')
        ind.set(qn('w:firstLine'), '0')
        pPr.append(ind)
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), 'left')
        pPr.append(jc)
        tabs = OxmlElement('w:tabs')
        tab = OxmlElement('w:tab')
        tab.set(qn('w:val'), 'right')
        tab.set(qn('w:leader'), 'dot')
        tab.set(qn('w:pos'), '9071')
        tabs.append(tab)
        pPr.append(tabs)
    p.append(pPr)
    
    # Run 1: Chapter title
    r1 = OxmlElement('w:r')
    if level == 1:
        rPr1 = OxmlElement('w:rPr')
        b1 = OxmlElement('w:b')
        bCs1 = OxmlElement('w:bCs')
        rPr1.append(b1)
        rPr1.append(bCs1)
        r1.append(rPr1)
    t1 = OxmlElement('w:t')
    t1.text = title
    r1.append(t1)
    p.append(r1)
    
    # Run 2: Tab with dot leader
    r2 = OxmlElement('w:r')
    tab_el = OxmlElement('w:tab')
    r2.append(tab_el)
    p.append(r2)
    
    # Run 3: Page number
    r3 = OxmlElement('w:r')
    if level == 1:
        rPr3 = OxmlElement('w:rPr')
        b3 = OxmlElement('w:b')
        bCs3 = OxmlElement('w:bCs')
        rPr3.append(b3)
        rPr3.append(bCs3)
        r3.append(rPr3)
    t3 = OxmlElement('w:t')
    t3.text = str(page)
    r3.append(t3)
    p.append(r3)
    
    return p

BASE_DIR = "/Users/igorcerovsky/Documents/finger/paper/james"
TEMPLATE_PATH = os.path.join(BASE_DIR, "1_ICI_titulka_požiadavky_na_form_úpravu.docx")
OUTPUT_PATH = os.path.join(BASE_DIR, "cerovsky_practical_training_guide_sk.docx")

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner padding for a table cell (in twips, 20 twips = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Set individual borders for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    borders = {'top': top, 'bottom': top if False else bottom, 'left': left, 'right': right}
    for b_name, b_val in borders.items():
        if b_val:
            node = OxmlElement(f'w:{b_name}')
            node.set(qn('w:val'), b_val.get('val', 'single'))
            node.set(qn('w:sz'), str(b_val.get('sz', 4)))
            node.set(qn('w:space'), '0')
            node.set(qn('w:color'), b_val.get('color', 'auto'))
            tcBorders.append(node)
        else:
            node = OxmlElement(f'w:{b_name}')
            node.set(qn('w:val'), 'none')
            tcBorders.append(node)
    tcPr.append(tcBorders)

def set_table_borders(table, color="CCCCCC", sz="4"):
    """Set standard grid borders on a table."""
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    for b in ['top', 'left', 'bottom', 'right', 'insideH']:
        border = OxmlElement(f'w:{b}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), sz)
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), color)
        tblBorders.append(border)
    border = OxmlElement('w:insideV')
    border.set(qn('w:val'), 'none')
    tblBorders.append(border)
    tblPr.append(tblBorders)

def make_callout_box(doc, title, paragraphs, fill_hex="F2F5F9", border_color="2B579A"):
    """Create a stylized callout box with an accent left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Cm(16.0)
    
    set_cell_background(cell, fill_hex)
    set_cell_margins(cell, top=120, bottom=120, left=200, right=160)
    
    # Left border thick, others none
    set_cell_borders(cell, left={'val': 'single', 'sz': 24, 'color': border_color})
    
    # Add title
    p0 = cell.paragraphs[0]
    p0.style = doc.styles['Normal']
    p0.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p0.paragraph_format.first_line_indent = Cm(0)
    p0.paragraph_format.line_spacing = 1.2
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(2)
    run_title = p0.add_run(title)
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(11)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    
    for idx, text in enumerate(paragraphs):
        p = cell.add_paragraph()
        p.style = doc.styles['Normal']
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2 if idx < len(paragraphs) - 1 else 0)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10.5)
        
    # Add tiny trailing buffer paragraph inside cell so Pages row height calculation never clips the last line
    p_pad = cell.add_paragraph()
    p_pad.style = doc.styles['Normal']
    p_pad.paragraph_format.first_line_indent = Cm(0)
    p_pad.paragraph_format.line_spacing = 1.0
    p_pad.paragraph_format.space_before = Pt(0)
    p_pad.paragraph_format.space_after = Pt(0)
    r_pad = p_pad.add_run(" ")
    r_pad.font.size = Pt(4)
        
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(2)

def format_run(run, font_name="Times New Roman", size_pt=12, bold=False, italic=False, color=None):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color

def add_body_paragraph(doc, text="", bold_prefix="", space_before=0, space_after=4, first_line_indent=0.9):
    p = doc.add_paragraph(style='Normal')
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.first_line_indent = Cm(first_line_indent)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        format_run(r_pre, bold=True)
    if text:
        r_text = p.add_run(text)
        format_run(r_text)
    return p

def _disable_heading_numbering(p):
    pPr = p._p.get_or_add_pPr()
    numPr = OxmlElement('w:numPr')
    ilvl = OxmlElement('w:ilvl')
    ilvl.set(qn('w:val'), '0')
    numId = OxmlElement('w:numId')
    numId.set(qn('w:val'), '0')
    numPr.append(ilvl)
    numPr.append(numId)
    pPr.append(numPr)

def add_heading1(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles['Heading 1']
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(0)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(16)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    _disable_heading_numbering(p)
    
    # clear existing runs and set clean run
    p.text = ""
    r = p.add_run(text)
    format_run(r, font_name="Times New Roman", size_pt=15, bold=True, color=RGBColor(0x1F, 0x38, 0x64))
    return p

def add_heading2(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles['Heading 2']
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(0)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(12)
    pf.space_after = Pt(4)
    pf.keep_with_next = True
    _disable_heading_numbering(p)
    
    p.text = ""
    r = p.add_run(text)
    format_run(r, font_name="Times New Roman", size_pt=13, bold=True, color=RGBColor(0x2F, 0x55, 0x97))
    return p

def add_heading3(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles['Heading 3']
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(0)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(8)
    pf.space_after = Pt(2)
    pf.keep_with_next = True
    _disable_heading_numbering(p)
    
    p.text = ""
    r = p.add_run(text)
    format_run(r, font_name="Times New Roman", size_pt=12, bold=True, color=RGBColor(0x33, 0x33, 0x33))
    return p

def add_bullet_item(doc, text, bold_prefix="", level=0):
    p = doc.add_paragraph()
    p.style = doc.styles['Normal']
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.35
    pf.space_before = Pt(1)
    pf.space_after = Pt(3)
    
    indent_cm = 0.9 + level * 0.5
    pf.left_indent = Cm(indent_cm)
    pf.first_line_indent = Cm(-0.4)
    
    bullet_symbol = "•  " if level == 0 else "–  "
    r_bullet = p.add_run(bullet_symbol)
    format_run(r_bullet, bold=True)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        format_run(r_pre, bold=True)
    if text:
        r_text = p.add_run(text)
        format_run(r_text)
    return p

def add_equation_box(doc, equation_text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.line_spacing = 1.3
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.first_line_indent = 0
    
    r = p.add_run(equation_text)
    format_run(r, font_name="Cambria Math", size_pt=11.5, italic=False)
    return p

def build_document():
    print(f"Loading template: {TEMPLATE_PATH}")
    doc = docx.Document(TEMPLATE_PATH)
    
    # Check section setup
    sec = doc.sections[0]
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    sec.header_distance = Cm(1.25)
    sec.footer_distance = Cm(1.25)
    sec.different_first_page_header_footer = True
    
    # Format Title Page (Paragraphs 0 to 20)
    # P0: Course title
    doc.paragraphs[0].text = "Kurz inštruktorov horolezectva 1. kvalifikačného stupňa"
    doc.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in doc.paragraphs[0].runs:
        format_run(r, size_pt=16, bold=True)
        
    # P1: Course specialization
    doc.paragraphs[1].text = "Lezenie na umelých stenách a lanové prekážky"
    doc.paragraphs[1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in doc.paragraphs[1].runs:
        format_run(r, size_pt=14, italic=True)
        
    # P2: Organization
    doc.paragraphs[2].text = "Slovenský horolezecký spolok JAMES, Horolezecká škola JAMES"
    doc.paragraphs[2].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in doc.paragraphs[2].runs:
        format_run(r, size_pt=15, bold=True)
        
    # P4: Záverečná práca
    doc.paragraphs[4].text = "Záverečná práca"
    doc.paragraphs[4].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in doc.paragraphs[4].runs:
        format_run(r, size_pt=20, bold=True)
        
    # P5: Názov práce a podnázov (P6 obsahuje vodoznak James_logo_1, nesmie sa prepísať!)
    doc.paragraphs[5].text = ""
    doc.paragraphs[5].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[5].paragraph_format.space_before = Pt(6)
    doc.paragraphs[5].paragraph_format.space_after = Pt(12)
    r_title = doc.paragraphs[5].add_run("Odporúčania pre tréning prstov na základe biomechanického modelovania morfotypu prsta\n")
    format_run(r_title, size_pt=20, bold=True, color=RGBColor(0x1F, 0x38, 0x64))
    r_sub = doc.paragraphs[5].add_run("Praktický sprievodca pre trénerov, inštruktorov a lezcov")
    format_run(r_sub, size_pt=14, italic=True, color=RGBColor(0x40, 0x40, 0x40))
        
    # P6: PONECHAŤ BEZ ZMENY! Obsahuje ukotvený vodoznak James_logo_1 (rId8 -> image1.png)
        
    # Place and date at P19
    doc.paragraphs[19].text = "Bratislava, 2026"
    doc.paragraphs[19].alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in doc.paragraphs[19].runs:
        format_run(r, size_pt=14)
        
    # Author at P20
    doc.paragraphs[20].text = "Mgr. Igor Cerovský PhD."
    doc.paragraphs[20].alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in doc.paragraphs[20].runs:
        format_run(r, size_pt=16, bold=True)
        
    # Remove 3 excess empty spacer paragraphs (P16..P18) so multi-line title + page break fit on Page 1
    body = doc._body._element
    for idx_to_del in [18, 17, 16]:
        p_del = doc.paragraphs[idx_to_del]._element
        body.remove(p_del)
        
    # Populate Table of Contents (sdt)
    sdt = doc._body._element.find(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sdt")
    content = sdt.find(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sdtContent")
    
    # Clear placeholder dummy entries from template
    for child in list(content):
        content.remove(child)
        
    content.append(make_toc_header())
    content.append(make_empty_p())
    
    toc_items = [
        ("Úvod: Prepojenie biomechaniky s lezeckou praxou", 3, 1),
        ("1. Základné biomechanické odporúčania pre tréning prstov", 4, 1),
        ("1.1 Dávkovanie úchopov a otvorený úchop ako ochrana pútok", 4, 2),
        ("1.2 Kolinearita predlaktia a optimalizácia ťahového vektora", 5, 2),
        ("1.3 Periodizácia tréningu pre lezcov s dlhými prstami", 6, 2),
        ("1.4 Práca nôh ako prevencia rázového preťaženia pútok", 7, 2),
        ("1.5 Rozdielna dĺžka prstov a ergonomický výber líšt", 8, 2),
        ("1.6 Ochrana kĺbového puzdra pri zošmyknutí ruky z chytu", 8, 2),
        ("1.7 Voľba hĺbky testovacej lišty podľa dĺžky prstov", 10, 2),
        ("1.8 Tribológia kože, teplota a práca s trením", 11, 2),
        ("2. Diagnostické sebahodnotenie preťaženia pútok a kĺbov", 12, 1),
        ("3. Biomechanické odporúčania podľa vekových kategórií", 13, 1),
        ("3.0 Zmena pákových pomerov počas rastu a vývoja", 13, 2),
        ("3.1 Deti (do 12–13 rokov / predpubertálne obdobie)", 13, 2),
        ("3.2 Mládež a juniori (13–18 rokov / rastový špurt)", 14, 2),
        ("3.3 Dospelí lezci (18–45 rokov / kostrová zrelosť)", 15, 2),
        ("3.4 Starší dospelí a kategória masters (nad 45 rokov)", 15, 2),
        ("3.5 Prehľad pákových pomerov a rizík podľa veku", 16, 2),
        ("4. Periodizácia podľa morfotypu prsta a návrh mikrocyklov", 17, 1),
        ("5. Stručný fyzikálny prehľad: Teoretické východiská modelu", 18, 1),
        ("Záver: Dlhodobo udržateľné zdravie ako predpoklad výkonnosti", 18, 1),
        ("Literatúra", 18, 1)
    ]
    
    for title, page, level in toc_items:
        content.append(make_toc_entry(title, page, level=level))
        
    # Enable automatic field updates on document open in Word
    update_fields = OxmlElement('w:updateFields')
    update_fields.set(qn('w:val'), 'true')
    doc.settings.element.append(update_fields)
    
    # Manage page breaks and remove template instructions
    sdt_idx = None
    for idx, el in enumerate(body):
        if el.tag.endswith('sdt'):
            sdt_idx = idx
            break
            
    # Page break after Table of Contents (sdt) (make_toc_header already has w:pageBreakBefore)
    ns_w = nsdecls('w')
    br_after = parse_xml('<w:p ' + ns_w + '><w:r><w:br w:type="page"/></w:r></w:p>')
    body.insert(sdt_idx + 1, br_after)
    
    # Remove old template instructions (everything between br_after and sectPr)
    sectPr = body.find(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr")
    elements_to_remove = []
    start_deleting = False
    for el in list(body):
        if el is br_after:
            start_deleting = True
            continue
        if el is sectPr:
            break
        if start_deleting:
            elements_to_remove.append(el)
            
    for el in elements_to_remove:
        body.remove(el)
        
    print("Table of Contents generated. Title page, Obsah (Page 2), and Body (Page 3+) cleanly partitioned.")
    
    # --- PAGE 3 ONWARDS: BODY CONTENT ---
    
    # Metadata info card at the top of Page 3
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_widths = [Cm(4.5), Cm(11.5)]
    meta_data = [
        ("Autor:", "Mgr. Igor Cerovský PhD. (Nezávislý výskumník v oblasti fyziky a biomechaniky lezenia)"),
        ("Dátum:", "21. október 2026"),
        ("Korešpondencia:", "igor.cerovsky@gmail.com"),
        ("Repozitár projektu:", "https://github.com/igorcerovsky/finger"),
        ("Sprievodná vedecká štúdia:", "A Three-Dimensional Musculoskeletal Model of the Human Finger in Rock Climbing: Contact Mechanics, Pulley Shearing, and Phenotypic Scaling (Cerovský, 14. október 2026)")
    ]
    for row_idx, (label, val) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        for c_idx, cell in enumerate(row.cells):
            cell.width = meta_widths[c_idx]
            set_cell_background(cell, "F9FBFD" if row_idx % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            set_cell_borders(cell, bottom={'val': 'single', 'sz': 2, 'color': 'E0E8F0'})
        p_lbl = row.cells[0].paragraphs[0]
        p_lbl.paragraph_format.first_line_indent = 0
        p_lbl.paragraph_format.line_spacing = 1.15
        r_lbl = p_lbl.add_run(label)
        format_run(r_lbl, size_pt=10, bold=True, color=RGBColor(0x33, 0x33, 0x33))
        
        p_val = row.cells[1].paragraphs[0]
        p_val.paragraph_format.first_line_indent = 0
        p_val.paragraph_format.line_spacing = 1.15
        r_val = p_val.add_run(val)
        format_run(r_val, size_pt=10, italic=(row_idx == 4))
        
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_before = Pt(8)
    p_div.paragraph_format.space_after = Pt(8)
    
    # SECTION: Úvod
    add_heading1(doc, "Úvod: Prepojenie biomechaniky s lezeckou praxou")
    add_body_paragraph(
        doc,
        "Metodika tréningu prstov v športovom lezení sa dlhé roky vyvíjala predovšetkým empiricky na základe skúseností lezcov a trénerov. Tento prístup však nezriedka viedol k preťaženiu, akútnym ruptúram pútok alebo chronickým zápalom šliach (tendovaginitídam). Hoci sa štandardizované protokoly na hangboardoch (napr. na 20 mm lište) a visy s pridanou hmotnosťou stali bežnou tréningovou praxou, často nezohľadňujú individuálne rozdiely a priestorovú mechaniku pohybového aparátu."
    )
    add_body_paragraph(
        doc,
        "Výsledky navrhnutého priestorového (3D) modelu ukazujú, že skutočné zaťaženie ohýbačov prstov závisí predovšetkým od rozloženia kontaktného tlaku na chyte, pomeru dĺžky článkov prsta k hĺbke lišty a priestorového smerovania ťahových síl. Cieľom tejto práce je pretaviť tieto biomechanické poznatky do ôsmich praktických odporúčaní, ktoré pomáhajú zvyšovať lezeckú výkonnosť a zároveň chrániť šľachy, pútka a kĺby pred dlhodobým preťažením."
    )
    
    # SECTION 1: Základné biomechanické odporúčania
    add_heading1(doc, "1. Základné biomechanické odporúčania pre tréning prstov")
    
    make_callout_box(
        doc,
        "Klasifikácia úrovne dôkazov:",
        [
            "Pre prehľadnosť a odbornú korektnosť je každé odporúčanie označené podľa hlavného zdroja podkladov:",
            "• [Odvodené z modelu]: Kvantitatívne výsledky získané priamo z navrhnutého 3D biomechanického modelu prsta a kontaktnej mechaniky.",
            "• [Podložené literatúrou]: Publikované empirické poznatky zo športovej medicíny, kadaveróznych meraní alebo tribológie.",
            "• [Mechanistická hypotéza]: Odborné predpoklady a trénerské odporúčania vychádzajúce z mechanických princípov, ktoré si vyžadujú ďalšie priame experimentálne overenie."
        ]
    )
    
    # Recommendation 1.1
    add_heading2(doc, "1.1 Dávkovanie úchopov a otvorený úchop ako ochrana pútok [Odvodené z modelu a podložené literatúrou]")
    add_body_paragraph(
        doc,
        "Pri zamknutom úchope (full crimp) spôsobuje hyperextenzia v DIP kĺbe to, že šľacha hlbokého ohýbača prstov (FDP) preberá až 65 % celkového zaťaženia, zatiaľ čo výrazné ohnutie v PIP kĺbe (105°) smeruje výslednicu sily šľachy kolmo proti pútku A2. Normálová sila pôsobiaca na pútko A2 tak pri štandardnom vise so záťažou 100 N na 10 mm lište dosahuje 267,0 N (a pri dynamickom zachytení so záťažou 171,7 N narastá až na 457,5 N). Naopak, pri otvorenom úchope (open hand) klesá normálová sila na pútko A2 na 62,6 N pri záťaži 100 N (resp. na 107,5 N pri dynamickom zaťažení), čo predstavuje zníženie namáhania pútka o viac než 76 % (pričom zaťaženie pútka zostáva v celom rozsahu bežných tréningových váh bezpečne pod hranicou 100 N).",
        bold_prefix="Biomechanický princíp: "
    )
    add_body_paragraph(doc, "Odporúčania pre tréningovú prax:", bold_prefix="")
    add_bullet_item(doc, "Približne 80 % celkového tréningového objemu (objemové lezenie, vytrvalostné intervaly, základné visy na hangboarde) sa odporúča absolvovať v otvorenom úchope (open hand) a v polouzamknutom úchope (half-crimp).", bold_prefix="Pravidlo 80 / 20: ")
    add_bullet_item(doc, "Zamknutý úchop (full crimp so zámkom palca) je vhodné vnímať ako špecifickú silovú rezervu. Odporúča sa vyhradiť ho predovšetkým na ostré pokusy v ťažkých cestách a bouldroch na lištách menších ako 8 mm, kde iný úchop neumožňuje udržať potrebný moment v DIP kĺbe, a nevyužívať ho pri tréningu do úplného svalového zlyhania.", bold_prefix="Uvážlivé využívanie zamknutého úchopu: ")

    # Recommendation 1.2
    add_heading2(doc, "1.2 Kolinearita predlaktia a optimalizácia ťahového vektora [Odvodené z modelu a technická heuristika]")
    add_body_paragraph(
        doc,
        "Anulárne pútka (A2, A4) predstavujú anizotropné väzivovo-chrupavkové oblúky prispôsobené na obvodové napätie (udržiavanie šliach pri kosti). V lokálnom anatomickom súradnicovom systéme proximálneho článku vykazuje stred pútka A2 pri čistej sagitálnej flexii minimálne ohybové zaťaženie mimo roviny (FA2,lat_local ≈ 0 N). Keď sa však MCP kĺb dostane do radiálnej abdukcie mimo sagitálnej roviny (15°), vzniká výrazný priečny presmerovací strih koncentrovaný na pútku A1 / vstupnom palmárnom ústí pútka (137,3 N pri zaťažení 171,7 N; 80,4 N pri zaťažení 100 N). To je sprevádzané 44,4 N mediolaterálneho strihu na kolaterálnych väzoch PIP kĺbu (25,9 N pri zaťažení 100 N) a globálnou laterálnou zložkou 93,0 N (54,5 N pri zaťažení 100 N) odtiahnutia šľachy od kosti (bowstringing) na pútku A2.",
        bold_prefix="Biomechanický princíp: "
    )
    add_body_paragraph(doc, "Lezecká technika – ťažisko tela (CoG) a optimálny „ťahový vektor“:", bold_prefix="")
    add_bullet_item(doc, "Efektivita lezeckej techniky zásadne závisí od polohy tela. Aktívnym riadením ťažiska (boky, trup a dynamická práca nôh) môžu lezci smerovať výsledný ťahový vektor čo najkolmejšie na využiteľný povrch chytu.", bold_prefix="Ťažisko určuje nároky na prsty: ")
    add_bullet_item(doc, "Optimalizácia ťahového vektora prostredníctvom práce tela znižuje celkovú silu zovretia prstov potrebnú na udržanie kroku (maximalizuje efektívne trenie a mechanickú páku), vďaka čomu ťažké kroky pôsobia kontrolovateľnejšie. Súčasne minimalizuje parazitné krútiace momenty mimo roviny, priečny strih na vstupe do pútka A1 a namáhanie kolaterálnych väzov, čím znižuje dôležitý mechanický spúšťač akútnych zranení tkanív. Vybrúsená technika tak slúži ako účinná ochrana pred zranením.", bold_prefix="Dvojitý prínos (výkon + prevencia): ")
    add_bullet_item(doc, "Odporúča sa vyhnúť prepadávaniu lakťa dovnútra pri bočných ťahoch (gastonoch) alebo jeho neprimeranému vytáčaniu von („kuracie krídla“) pri zaťažení bočných líšt. Takéto zalomenie rotuje šľachovú pošvu a vytvára zvýšenú koncentráciu strihového napätia na bočných okrajoch pútok.", bold_prefix="Pasca polohy lakťov: ")
    add_body_paragraph(doc, "Bezpečnosť tréningu na hangboarde a so silomermi (Tindeq):", bold_prefix="")
    add_bullet_item(doc, "Pri testovaní alebo tréningu s izometrickými senzormi sily (napr. Tindeq Progressor, závesné váhy či inteligentné dosky) môžu lezci podvedome vytáčať zápästie, rozťahovať lakte alebo krútiť prsty, aby dosiahli vyššiu špičkovú nameranú hodnotu. Tým však vzniká nežiaduci kompenzačný vzorec: páčenie kĺbov a zapájanie svalov ramena a chrbta na úkor asymetrického strihu na vstupe do pútka A1 a preťaženia kolaterálnych väzov.", bold_prefix="Pasca honby za „maximálnym číslom“: ")
    add_bullet_item(doc, "Na hangboardoch alebo prenosných lištách sa odporúča polohovať prsty kolmo na hranu, aby sa minimalizovalo krútenie. Namiesto maximalizácie nominálneho čísla na senzore je vhodné uprednostniť čistú sagitálnu mechaniku, tréningovú efektivitu a dlhodobé zdravie kĺbov. Pri sledovaní progresu je navyše oveľa dôležitejší dlhodobý trend (pri zachovaní rovnakej techniky úchopu) než samotná absolútna nameraná hodnota.", bold_prefix="Trénovať pre efektívnosť, nie pre falošné rekordy: ")

    # Recommendation 1.3
    add_heading2(doc, "1.3 Periodizácia tréningu pre lezcov s dlhými prstami [Odvodené z modelu a periodizačná hypotéza]")
    add_body_paragraph(
        doc,
        "Pri realistickom sublineárnom alometrickom škálovaní kĺbových kondylov (ma ∝ L^0,50) majú lezci s dlhými prstami na mikro-lištách prirodzenú pákovú nevýhodu +17,7 % (297,3 N oproti 252,6 N na 6 mm lište pri rovnakej vonkajšej sile). Pri lezení s vlastnou hmotnosťou sa k tomu pridáva nárast telesnej hmotnosti úmerný výške postavy (m ∝ L^2–L^3), v dôsledku čoho stúpa zaťaženie pútka A2 na 10 mm lište zo 174,2 N (krátke prsty) na 380,0 N (dlhé prsty). Tento nárast o +118,1 % sa už približuje k medznej pevnosti pútka známej z kadaveróznych meraní (~400 N). Plytké lišty navyše poskytujú dlhšiemu koncovému článku menšie relatívne skrátenie páky (ρ = d / L_DP) a poskladanie dlhších článkov na malý chyt si vyžaduje ostrejší uhol v PIP kĺbe (θ_PIP > 100°), čo ešte viac zvyšuje ťah šľachy do pútka A2.",
        bold_prefix="Biomechanický princíp: "
    )
    add_body_paragraph(doc, "Odporúčania pre tréning lezcov s dlhými prstami:", bold_prefix="")
    add_bullet_item(doc, "Kým kostrové svaly sa prispôsobujú tréningovému zaťaženiu v priebehu týždňov, prestavba kolagénu a zosilňovanie šľachových pútok prebiehajú výrazne pomalšie (v praxi sa počíta s horizontom 12 až 24 mesiacov postupnej adaptácie). Lezcom s dlhšími prstami sa preto pri prechode na menšie lišty a pri pridávaní závažia odporúča postupovať opatrnejšie než lezcom s kratšími prstami.", bold_prefix="Pozvoľnejšia progresia pre spojivové tkanivá: ")
    add_bullet_item(doc, "Keďže pri lezení s vlastnou hmotnosťou vzniká v dlhších prstoch vyššie absolútne napätie v pútku aj vyšší tlak v kĺboch, medzi vysoko intenzívnymi tréningami prstov sa odporúča ponechať 48 až 72 hodín na regeneráciu spojivových tkanív.", bold_prefix="Dlhší čas na regeneráciu: ")
    add_bullet_item(doc, "Lezci s dlhými prstami majú prirodzenú výhodu na oblinách, stiskoch, väčších štruktúrach a v kompresných krokoch, kde dokážu využiť väčšiu kontaktnú plochu a obopnutie chytu. Naopak, na mikro-lištách sú v mechanickej nevýhode; namiesto silenia zamknutého úchopu (full crimp) sa im odporúča klásť dôraz na presnú prácu nôh (vysoké stupy, odľahčenie cez boky, drop-knee) a využívanie otvoreného úchopu.", bold_prefix="Využitie morfologických predností: ")

    # Recommendation 1.4
    add_heading2(doc, "1.4 Práca nôh ako prevencia rázového preťaženia pútok [Podložené literatúrou a biomechanická extrapolácia]")
    add_body_paragraph(
        doc,
        "V statickej rovnováhe na 10 mm lište pôsobí na pútko A2 pri zaťažení prostredníka silou 100 N sila v rozmedzí 265,1–267,0 N, čo je v rámci fyziologickej tolerancie tkaniva. Ak však noha nečakane skĺzne zo stupu, vonkajšie zaťaženie prsta skokovo narastie na 25–40 % telesnej hmotnosti (171,7 N). Pri takomto ráze stúpne normálová sila na pútko A2 pri štandardnom morfotype na 457,5 N, čo výrazne prekračuje medznú pevnosť pútka zistenú pri kadaveróznych meraniach (~400 N; Lin et al. 1990; Schöffl et al. 2009).",
        bold_prefix="Biomechanický princíp: "
    )
    add_body_paragraph(doc, "Odporúčania pre tréningovú prax:", bold_prefix="")
    add_bullet_item(doc, "Ak nohy nečakane vypadnú zo stupov, odporúča sa chyt radšej vedome pustiť a odskočiť, než sa ho snažiť udržať kŕčovitým stiskom za každú cenu. Akútne ruptúry pútok IV. stupňa vznikajú veľmi často práve pri náhlom excentrickom ráze po šmyku nohy.", bold_prefix="Návyk uvoľniť úchop pri šmyku nohy: ")
    add_bullet_item(doc, "Kvalitná práca nôh a spevnenie stredu tela (core) priamo chránia pútka ohýbačov prstov. Cielený nácvik presného ukladania nôh na vyšmýkané stupy znižuje riziko nečakaného šmyku v momentoch, keď sú prsty maximálne zaťažené.", bold_prefix="Dôraz na presnú prácu nôh a spevnenie trupu: ")
    add_bullet_item(doc, "Podrážky lezečiek sa odporúča udržiavať čisté. Prach, nánosy magnézia a nečistoty znižujú koeficient trenia (μ), a to najmä na vyšmýkanom vápenci alebo na hladkých stupoch (no-tex / dual-tex) na umelých stenách. Očistenie lezečiek pred ťažkým pokusom je jednoduchou prevenciou náhleho šmyku nôh a následného rázového preťaženia pútok.", bold_prefix="Čistota lezečiek: ")

    # Recommendation 1.5
    add_heading2(doc, "1.5 Rozdielna dĺžka prstov a ergonomický výber líšt [Klinická heuristika podložená literatúrou]")
    add_body_paragraph(
        doc,
        "Prostredník (III. prst) je výrazne dlhší než susedný ukazovák (II. prst) a prstenník (IV. prst). Pri úchope na rovnej vodorovnej lište sa preto končeky prstov prirodzene zrovnávajú do jednej línie, čo núti prostredník do nadmerného zohnutia (hyperflexie >105°) v PIP kĺbe a sústreďuje na jeho pútko A2 neúmerne vysokú normálovú silu a krútiaci moment. Navyše, keďže hlboké ohýbače III., IV. a V. prsta vychádzajú zo spoločného svalového bruška (tzv. efekt quadriga), rozdielne vystretie susedných prstov vyvoláva výrazné strihové napätie medzi šľachami a v oblasti lumbrikálnych svalov.",
        bold_prefix="Biomechanický princíp: "
    )
    add_body_paragraph(doc, "Odporúčania pre výber pomôcok a tréning:", bold_prefix="")
    add_bullet_item(doc, "Pri tréningu na hangboarde sa odporúča uprednostniť mierne zaoblené, odstupňované alebo skosené lišty, ktoré rešpektujú prirodzený rozdiel v dĺžke prstov. Záťaž sa tak rozloží rovnomernejšie medzi všetky štyri prsty bez toho, aby bol prostredník nútený do extrémneho ohybu v PIP kĺbe.", bold_prefix="Ergonomicky tvarované lišty: ")
    add_bullet_item(doc, "Pri zaťažovaní dierok dvojicou prstov (najčastejšie prostredník a prstenník) sa odporúča nekrčiť voľný ukazovák a malíček kŕčovito do dlane. Ak nezaťažené prsty zostanú mierne vystreté a uvoľnené, znižuje sa vzájomný posun šliach hlbokého ohýbača a chránia sa lumbrikálne svaly pred natrhnutím.", bold_prefix="Zásady bezpečnosti pri tréningu na dierkach pre dva prsty: ")

    # Recommendation 1.6
    add_heading2(doc, "1.6 Ochrana kĺbového puzdra pri zošmyknutí ruky z chytu [Klinické pozorovanie a mechanistická hypotéza]")
    add_body_paragraph(doc, "Mechanizmus postupného zošmykovania prstov:", bold_prefix="")
    add_bullet_item(doc, "Na ľudskej ruke presahuje prostredník (III. prst) susedný ukazovák (II. prst) a prstenník (IV. prst) zvyčajne o 8 až 15 mm.", bold_prefix="Prečnievanie prostredníka: ")
    add_bullet_item(doc, "Keď sa ruka začne nečakane šmýkať z chytu (vplyvom potu, vyšmýkanej skaly, hladkých dual-tex chytov alebo prudkého dynamického kroku), prsty sa z chytu neuvoľnia súčasne. Kratší ukazovák a prstenník skĺznu z hrany ako prvé.", bold_prefix="Postupná strata kontaktu pri šmyku: ")
    add_bullet_item(doc, "Počas kritických 20–60 milisekúnd pri zošmyknutí sa celá hybnosť padajúceho tela prenesie na jediný prst, ktorý zostal v kontakte s chytom – prečnievajúci prostredník.", bold_prefix="Sústredenie celého impulzu na jeden prst: ")
    add_bullet_item(doc, "Keď sa prostredník pod plnou hmotnosťou tela ťahá cez ostrú hranu chytu, dochádza k jeho prudkému, nútenému vystretiu (excentrickej extenzii). Keďže tvar chytu a smer ťahu sú len málokedy presne kolmé, na prst zároveň pôsobí výrazný krútiaci (torzný) moment (M = r × F_slip).", bold_prefix="Prudké nútené vystretie a krútenie (torzia): ")
    add_bullet_item(doc, "Lezci s dlhými prstami majú podstatne dlhšie páky stredného a koncového článku (L_MP + L_DP), čo priamo znásobuje vonkajší krútiaci moment v PIP kĺbe a vystavuje kĺbové puzdro vysokým špičkovým torzným silám.", bold_prefix="Väčšie pákové rameno pri dlhých prstoch: ")
    
    add_body_paragraph(doc, "Mechanizmus poškodenia kĺbového puzdra a palmárnej doštičky:", bold_prefix="")
    add_bullet_item(doc, "Zatiaľ čo anulárne pútka zlyhávajú najmä pri ťahu šľachy smerom od kosti (bowstringing), väzivové puzdro PIP kĺbu, kolaterálne väzy a väzivovo-chrupavková palmárna (volárna) doštička znášajú hlavný nápor pri násilnom vystretí a bočnom rotačnom strihu.")
    add_bullet_item(doc, "Takéto trhnutie spôsobuje mikrotrhliny v úponoch kolaterálnych väzov a preťažuje úpon palmárnej doštičky, čo vedie k akútnemu alebo chronickému zápalu kĺbového puzdra a výstelky (kapsulitíde / synovitíde PIP kĺbu).")
    add_bullet_item(doc, "Na rozdiel od akútneho prasknutia pútka sa poranenie kĺbového puzdra prejavuje tupou, difúznou bolesťou po celom obvode PIP kĺbu („opuchnutý, zhrubnutý kĺb“), citlivosťou pri stlačení kĺbu zboku, rannou stuhnutosťou a ostrou bolesťou pri náhodnom bočnom buchnutí alebo vytočení prsta.", bold_prefix="Klinické prejavy: ")

    add_body_paragraph(doc, "Praktická prevencia a režimové opatrenia:", bold_prefix="")
    add_bullet_item(doc, "Odporúča sa vedome trénovať reakciu na pošmyknutie ruky – ak prsty začnú z chytu nekontrolovane schádzať, je bezpečnejšie úchop včas uvoľniť, než nechať celú hmotnosť tela prejsť cez jediný ostávajúci prostredník.", bold_prefix="Potlačenie reflexu kŕčovitého zachytávania šmyku: ")
    add_bullet_item(doc, "Lezcom s výrazne dlhším prostredníkom alebo s opakovaným podráždením kĺbového puzdra sa odporúča podporné X-tejpovanie na dlaňovej strane PIP kĺbu. Prekrížené pruhy tejpu mechanicky brzdia násilné vystretie a bočný výkyv kĺbu bez toho, aby obmedzovali bežný ohyb prsta.", bold_prefix="Funkčné tejpovanie PIP kĺbu do kríža (X-tejpovanie): ")
    add_bullet_item(doc, "Pri boulderingu na vyšmýkaných alebo dynamických oblinách poskytuje voľné zatejpovanie prostredníka s prstenníkom (III. a IV. prst spojené páskou v oblasti základného a stredného článku) účinnú bočnú oporu a pri prípadnom šmyku rozloží zaťaženie na oba prsty.", bold_prefix="Spojenie susedných prstov (buddy-taping) pri rizikovom lezení: ")
    add_bullet_item(doc, "", bold_prefix="Postup pri podráždení a rehabilitácii synovitídy:")
    add_bullet_item(doc, "Tlačenie opuchnutého kĺbu so zapáleným puzdrom do pasívneho vystretia či úplného zohnutia zhoršuje dráždenie synoviálnej výstelky.", bold_prefix="Vynechanie násilného pasívneho naťahovania: ", level=1)
    add_bullet_item(doc, "Vhodné sú mierne, submaximálne izometrické zaťaženia v polouzamknutom úchope (half-crimp) alebo otvorenom úchope pri ohnutí kĺbu o 45°–60°, ktoré podporujú prekrvenie, pohyb kĺbovej tekutiny a prestavbu kolagénu.", bold_prefix="Aktívna izometria v strednom rozsahu ohybu: ", level=1)
    add_bullet_item(doc, "Striedavé ponáranie ruky do teplej a studenej vody pomáha zmierniť pretrvávajúci opuch kĺbu po záťaži a podporuje látkovú výmenu v málo prekrvenom tkanive kĺbového puzdra.", bold_prefix="Striedavé kúpele (kontrastná hydroterapia): ", level=1)

    # Recommendation 1.7
    add_heading2(doc, "1.7 Voľba hĺbky testovacej lišty podľa dĺžky prstov [Hypotéza odvodená z modelu]")
    add_body_paragraph(
        doc,
        "Bežné testovacie protokoly hodnotia všetkých lezcov na rovnakej 20 mm lište. Z biomechanického hľadiska sa však skutočná (relatívna) hĺbka chytu mení v závislosti od dĺžky koncového článku prsta (d / L_DP):",
        bold_prefix="Úskalia jednotnej 20 mm testovacej lišty: "
    )
    add_bullet_item(doc, "Pre morfotyp s výrazne krátkymi prstami (spodné percentily populácie, L_DP = 18,7 mm) predstavuje 20 mm lišta hlboký chyt (d / L_DP = 1,07). Do opory o chyt sa zapája aj stredný článok, časť zaťaženia preberá pútko A3 a nároky na šľachu hlbokého ohýbača (FDP) výrazne klesajú.")
    add_bullet_item(doc, "Pre morfotyp s výrazne dlhými prstami (horné percentily populácie, L_DP = 25,3 mm) predstavuje rovnaká 20 mm lišta plytkú lištu (d / L_DP = 0,79), na ktorej celé zaťaženie nesie iba koncový článok, čo podstatne zvyšuje nároky na ohýbače prstov.")
    add_body_paragraph(doc, "Praktické odporúčanie pre diagnostiku a testovanie:", bold_prefix="")
    add_body_paragraph(doc, "Pri porovnávacom testovaní sa odporúča prispôsobiť hĺbku testovacej lišty anatómii prsta konkrétneho lezca podľa vzťahu:")
    add_equation_box(doc, "d_test = 0,80 × L_DP")
    add_bullet_item(doc, "Pre štandardne dlhý prst (L_DP ≈ 22 mm) tomu zodpovedá testovacia lišta s hĺbkou 17,5–18 mm.")
    add_bullet_item(doc, "V krajných prípadoch morfotypu vychádza pre krátke prsty (18,7 mm) ekvivalentná hĺbka 15 mm, zatiaľ čo pre dlhé prsty (25,3 mm) hĺbka 20–22 mm.")

    # Recommendation 1.8
    add_heading2(doc, "1.8 Tribológia kože, teplota a práca s trením [Podložené literatúrou a tribologická heuristika]")
    add_body_paragraph(
        doc,
        "Trenie medzi kožou bruška prsta a povrchom skaly či chytu má nelineárny charakter a pri vysokom kontaktnom tlaku relatívne klesá (μ ∝ F_N^(n-1)). Rohová vrstva pokožky (stratum corneum) sa správa ako viskoelastický materiál, pričom koeficient trenia v závislosti od teploty a vlhkosti kože opisuje krivku v tvare obráteného písmena U:",
        bold_prefix="Biomechanický princíp: "
    )
    add_bullet_item(doc, "Pri vyššej teplote okolia (>20 °C) alebo nadmernom potení prstov vzniká na povrchu kože tenký tekutý film, ktorý znižuje koeficient trenia μ z hodnôt okolo 0,65 pod 0,35. Lezec tento pokles trenia podvedome kompenzuje silnejším stiskom chytu (vyššou normálovou silou), čo priamo zvyšuje ťah v šľachách a zaťaženie pútok.", bold_prefix="Vysoká teplota a vlhkosť („šmýkanie na pote“, angl. greasing out): ")
    add_bullet_item(doc, "Naopak, silný chlad (<5 °C), nízka vlhkosť vzduchu (<30 % RH) alebo nadmerné vrstvenie magnézia spôsobujú opačný, rovnako rizikový jav označovaný ako dry-firing. V chladnom a suchom prostredí stráca povrchová vrstva kože svoju pružnosť a stáva sa tvrdou a krehkou. Bez prirodzenej vlhkosti sa pokožka nedokáže mikroskopicky prispôsobiť nerovnostiam chytu, prsty z neho bez varovania „ustrelia“ a náhle odľahčenie či zachytenie šmyku prenesie prudký ráz do pútok a kĺbových puzdier.", bold_prefix="Nízka teplota a presušená koža („ustrelenie na sucho“, angl. dry-firing): ")
    add_body_paragraph(doc, "Praktické odporúčania pre starostlivosť o kožu a prácu s trením:", bold_prefix="")
    add_bullet_item(doc, "Najlepšie trenie pokožka dosahuje v chladnejšom prostredí (≈ 8–15 °C pri relatívnej vlhkosti vzduchu 40–60 %), keď je dostatočne pevná v strihu, no zároveň si zachováva pružnosť potrebnú na kopírovanie mikroštruktúry skaly.", bold_prefix="Optimálne podmienky pre trenie: ")
    add_bullet_item(doc, "V chladnom a suchom počasí sa odporúča nepreháňať množstvo magnézia na už vysušených prstoch. Pred pokusom je vhodné zohriať prsty vo vreckách a v prípade príliš suchej kože na brušká prstov jemne dýchnuť – minimálna vlhkosť z dychu vráti koži poddajnosť bez toho, aby sa prsty spotili.", bold_prefix="Prevencia „ustrelenia na sucho“ (dry-firing): ")
    add_bullet_item(doc, "Pravidelné čistenie chytov kefkou odstraňuje nánosy mastnoty, gumy a udupaného magnézia a obnovuje drsnosť povrchu. Vhodná kombinácia tekutého magnézia ako základnej vrstvy a primeraného množstva sypkého magnézia (uhličitanu horečnatého) udrží prsty suché bez nadmerného presušenia pokožky.", bold_prefix="Čistenie chytov a používanie magnézia: ")

    # SECTION 2: Diagnostické sebahodnotenie
    add_heading1(doc, "2. Diagnostické sebahodnotenie preťaženia pútok a kĺbov")
    add_body_paragraph(
        doc,
        "Nasledujúci prehľad slúži na včasné rozpoznanie varovných príznakov mikrotraumatizácie tkanív ešte pred vznikom vážneho akútneho zranenia:"
    )
    
    # Table 1: Diagnostické sebahodnotenie
    t1_caption = doc.add_paragraph()
    t1_caption.paragraph_format.space_before = Pt(6)
    t1_caption.paragraph_format.space_after = Pt(3)
    t1_caption.paragraph_format.first_line_indent = 0
    r_cap1 = t1_caption.add_run("Tabuľka 1: Diagnostické sebahodnotenie preťaženia pútok a kĺbových puzdier")
    format_run(r_cap1, size_pt=10.5, bold=True, italic=True)
    
    t1 = doc.add_table(rows=6, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1, color="B0C4DE", sz="4")
    t1_widths = [Cm(3.2), Cm(2.8), Cm(4.8), Cm(5.2)]
    
    t1_headers = ["Varovný príznak", "Postihnutá štruktúra", "Biomechanická príčina", "Odporúčané opatrenie"]
    t1_data = [
        (
            "Ranná stuhnutosť na dlaňovej strane základného článku",
            "Pútko A2 a šľachová pošva",
            "Opakované vysoké obvodové napätie pri častom lezení po lištách",
            "Znížiť objem lezenia na malých lištách o 50 %; na 10 dní prejsť na otvorený úchop."
        ),
        (
            "Ostrá bolesť pri tlaku na dlaňovú stranu PIP kĺbu",
            "Distálny okraj pútka A2 / pútko A3",
            "Odťahovanie pútka od kosti pri hyperextenzii v DIP kĺbe v zamknutom úchope (full crimp)",
            "Okamžite vynechať zamknutý úchop; odľahčiť pútko H-tejpovaním alebo podporným prstencom."
        ),
        (
            "Bolesť pri ťahu za bočné chyty a gastony",
            "Vstup pútka A1 / bočné okraje pútok / kolaterálne väzy",
            "Nadmerný bočný strih pri odklone prsta v MCP kĺbe (FA1,lat > 80 N, FPIP,ML > 40 N)",
            "Upraviť polohu lakťa a predlaktia; nevytáčať zápästie a prsty na chyte do strán."
        ),
        (
            "Bolesť v dlani medzi prostredníkom a prstenníkom",
            "Lumbrikálny sval",
            "Efekt quadriga pri zaťažení dierok so zovretými voľnými prstami v dlani",
            "Prerušiť tréning na dvojprstových dierkach; trénovať len so všetkými štyrmi prstami v otvorenom úchope."
        ),
        (
            "Tupá bolesť po obvode PIP kĺbu a citlivosť zboku po šmyku ruky",
            "Kĺbové puzdro PIP, palmárna doštička a kolaterálne väzy",
            "Postupné zošmyknutie prstov z chytu, pri ktorom dlhší prostredník zachytil celú hybnosť pádu v prudkom vystretí a skrútení.",
            "Nekŕčovať ruku pri šmyku; použiť X-tejpovanie PIP kĺbu alebo spojenie III. a IV. prsta (buddy-taping); zaradiť jemnú izometriu v strednom ohybe a vyhnúť sa pasívnemu naťahovaniu do ústupu opuchu."
        )
    ]
    
    # Header row
    hdr_row = t1.rows[0]
    hdr_trPr = hdr_row._tr.get_or_add_trPr()
    hdr_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    hdr_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    for idx, text in enumerate(t1_headers):
        c = hdr_row.cells[idx]
        c.width = t1_widths[idx]
        set_cell_background(c, "D9E2F3")
        set_cell_margins(c, top=100, bottom=100, left=120, right=120)
        p = c.paragraphs[0]
        p.paragraph_format.first_line_indent = 0
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        format_run(r, size_pt=10, bold=True, color=RGBColor(0x1F, 0x38, 0x64))
        
    for r_idx, row_data in enumerate(t1_data):
        row = t1.rows[r_idx + 1]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        bg_col = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, text in enumerate(row_data):
            c = row.cells[c_idx]
            c.width = t1_widths[c_idx]
            set_cell_background(c, bg_col)
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            p.paragraph_format.first_line_indent = 0
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            format_run(r, size_pt=9.5, bold=(c_idx == 0))
            
    doc.add_paragraph() # Spacing

    # SECTION 3: Vekové kategórie
    add_heading1(doc, "3. Biomechanické odporúčania podľa vekových kategórií")
    
    add_heading2(doc, "3.0 Zmena pákových pomerov počas rastu a vývoja")
    add_body_paragraph(
        doc,
        "Navrhnutý 3D biomechanický model ukazuje, že vnútorné napätie v šľachách ohýbačov (T_FDP, T_FDS), reakčné sily v kĺboch (F_joint) a normálové sily pôsobiace na anulárne pútka (F_A2, F_A4) sú určené momentovou rovnováhou v jednotlivých kĺboch prsta:"
    )
    add_equation_box(doc, "M_joint = r_ext(L_DP, L_MP, L_PP) × F_ext")
    add_equation_box(doc, "T_FDP ≈ M_PIP / h_FDP ,        F_pulley = 2 T_FDP · sin(θ / 2)")
    add_body_paragraph(
        doc,
        "Keďže vnútorné rameno sily šľachy v osteofibróznom kanáli prsta rastie s hrúbkou kostry len sublineárne (ma ∝ L^0,50), ťah v šľachách a normálové zaťaženie pútok priamo narastajú s predlžovaním článkov prsta (L_DP, L_MP, L_PP)."
    )
    add_body_paragraph(
        doc,
        "Rozloženie tlaku pozdĺž prsta zároveň závisí od bezrozmerného pomeru hĺbky chytu k dĺžke koncového článku:"
    )
    add_equation_box(doc, "ρ = d / L_DP")
    add_body_paragraph(
        doc,
        "Ak je ρ ≤ 1,0, zaťaženie nesie výlučne koncový článok (jednočlánkový kontakt). Ak je ρ > 1,0, do opory o chyt sa zapája aj stredný článok (dvojčlánkový kontakt), ktorý preberá časť normálovej sily, stabilizuje pútko A3 a znižuje vonkajší ohybový moment v DIP kĺbe takmer na nulu."
    )

    add_heading2(doc, "3.1 Deti (do 12–13 rokov / predpubertálne obdobie)")
    add_body_paragraph(doc, "Dĺžka článkov prsta a relatívna hĺbka chytu (d / L_DP):", bold_prefix="")
    add_bullet_item(doc, "Články detských prstov sú výrazne kratšie: L_DP ≈ 13–16 mm pri celkovej dĺžke prsta L_tot ≈ 50–65 mm (oproti 22 mm a 95 mm u dospelého).")
    add_bullet_item(doc, "Na štandardných lištách v lezeckých centrách (18–22 mm) dosahuje pomer hĺbky chytu u dieťaťa hodnotu ρ = d / L_DP > 1,2–1,5. Chyt, ktorý je pre dospelého lištou, tak pre dieťa predstavuje dobrý chyt, na ktorom sa o povrch chytu prirodzene opiera aj stredný článok.", bold_prefix="Výhoda na bežných chytoch pre dospelých: ")
    add_bullet_item(doc, "Ak však stavitelia ciest na tréningu alebo mládežníckych pretekoch použijú plytké lišty (6–10 mm), pomer klesá na ρ ≤ 0,6 a celé zaťaženie sa sústredí na koncový článok prsta.", bold_prefix="Riziko detských mikro-líšt: ")
    add_body_paragraph(doc, "Náchylnosť rastových platničiek na pákové zaťaženie (PPSI / epifyziolýza):", bold_prefix="")
    add_bullet_item(doc, "Rastová platnička (fýza) na báze stredného článku je tvorená neosifikovanou chrupavkou, ktorá predstavuje mechanicky najzraniteľnejšie miesto detskej ruky.")
    add_bullet_item(doc, "Pri zamknutom úchope (full crimp) vyvoláva kombinácia hyperextenzie v DIP kĺbe a ostrého ohybu v PIP kĺbe vysoké strihové napätie priamo v neuzavretej rastovej štrbine. Opakované preťažovanie môže viesť k únavovým zlomeninám a poškodeniu rastovej platničky na báze stredného článku (Primary Periphyseal Stress Injury – PPSI, stupne 1–5; Schöffl et al., 2025).")
    add_body_paragraph(doc, "Odporúčania pre tréningovú prax:", bold_prefix="")
    add_bullet_item(doc, "U predpubertálnych detí sa na ochranu neuzavretých rastových platničiek neodporúča zaraďovať izolované visy na hangboarde, cvičenia na campuse (campus boarde) ani lezenie s pridanou záťažou.", bold_prefix="Vynechanie izolovaných visov na hangboarde a mikro-líšt: ")
    add_bullet_item(doc, "V tréningu detí sa odporúča uprednostniť väčšie ergonomické chyty (d ≥ 20 mm), obliny, štruktúry a stisky, pri ktorých sa o chyt opiera aj stredný článok (ρ > 1,0).", bold_prefix="Voľba chytov s oporou dvoch článkov (d ≥ L_DP): ")

    add_heading2(doc, "3.2 Mládež a juniori (13–18 rokov / rastový špurt)")
    add_body_paragraph(doc, "Rýchle predlžovanie pák v období vrcholu rastového špurtu (PHV):", bold_prefix="")
    add_bullet_item(doc, "Počas obdobia najrýchlejšieho rastu do výšky (Peak Height Velocity – PHV) sa články prstov predlžujú rýchlejšie, než sa dokážu prispôsobiť a spevnieť šľachy, pútka a rastové platničky.")
    add_bullet_item(doc, "Štúdie zo športovej medicíny (Schöffl et al., 2023, 2025) ukazujú, že akútne ruptúry anulárnych pútok sú u dospievajúcich lezcov pomerne zriedkavé; najčastejším miestom poškodenia zostáva ešte neuzavretá rastová platnička (fýza) na báze stredného článku (klasifikácia PPSI, stupne 1–5).", bold_prefix="Hlavné riziko – stresové poranenia rastovej platničky (PPSI): ")
    add_body_paragraph(doc, "Odporúčania pre tréningovú prax:", bold_prefix="")
    add_bullet_item(doc, "Trénerom a rodičom sa odporúča priebežne sledovať prírastky telesnej výšky a dĺžky ruky. V období prudkého rastu (>6–8 cm za rok) je vhodné dočasne znížiť objem lezenia na malých lištách a v zamknutom úchope.", bold_prefix="Sledovanie rýchlosti rastu: ")
    add_bullet_item(doc, "Ak sa u staršej mládeže zaraďuje tréning na hangboarde, odporúčajú sa visy s vlastnou hmotnosťou na dostatočne hlbokých lištách (d ≥ 20–25 mm, ρ ≥ 1,0) v otvorenom úchope alebo v polouzamknutom úchope (half-crimp); dynamické skoky na campuse je vhodné odložiť až do ukončenia kostrového rastu.", bold_prefix="Opatrný prístup k hangboardu na hlbších lištách (d ≥ 20 mm): ")

    add_heading2(doc, "3.3 Dospelí lezci (18–45 rokov / kostrová zrelosť)")
    add_body_paragraph(doc, "Ustálená dĺžka článkov a rozdiely medzi morfotypmi (L_tot ∈ [80, 110 mm]):", bold_prefix="")
    add_bullet_item(doc, "Pri sublineárnom alometrickom škálovaní kĺbov (ma ∝ L^0,50) majú lezci s dlhšími prstami na mikro-lištách prirodzenú pákovú nevýhodu +17,7 % (297,3 N oproti 252,6 N na 6 mm lište pri rovnakom zaťažení 100 N).")
    add_bullet_item(doc, "Pre dospelého lezca s krátkymi prstami (L_DP = 18,7 mm, ρ = 1,07) predstavuje 20 mm lišta dvojčlánkový chyt so zaťažením pútka A2 na úrovni 218,0 N, zatiaľ čo pre lezca s dlhými prstami (L_DP = 25,3 mm, ρ = 0,79) zostáva jednočlánkovou lištou so zaťažením pútka A2 až 263,6 N (nárast zaťaženia o +20,9 %).", bold_prefix="Rozdielny kontakt na štandardnej 20 mm testovacej lište: ")
    add_bullet_item(doc, "Pri zohľadnení rastu hmotnosti s výškou postavy (m ∝ L^2) pôsobí na pútko A2 pri vise na 10 mm lište u lezca s dlhými prstami (92,5 kg) sila 380,0 N, oproti 174,2 N u lezca s krátkymi prstami (50,6 kg) – ide o +118,1 % nárast absolútneho zaťaženia, ktorý sa blíži k medznej pevnosti pútka (~400 N).", bold_prefix="Vplyv vyššej telesnej hmotnosti na malých lištách (10 mm): ")
    add_body_paragraph(doc, "Odporúčania pre tréningovú prax:", bold_prefix="")
    add_bullet_item(doc, "Lezcom s dlhými prstami sa medzi náročnými tréningami na malých lištách odporúča dopriať 48 až 72 hodín odpočinku.", bold_prefix="Prispôsobenie objemu a regenerácie morfotypu: ")
    add_bullet_item(doc, "Voľba hĺbky testovacej lišty podľa dĺžky koncového článku zabezpečuje biomechanicky porovnateľné podmienky pri testovaní sily prstov.", bold_prefix="Individuálne prispôsobenie testovacej lišty (d_test ≈ 0,80 L_DP): ")

    add_heading2(doc, "3.4 Starší dospelí a kategória masters (nad 45 rokov)")
    add_body_paragraph(doc, "Dlhodobé kumulatívne zaťaženie kĺbov a opotrebovanie chrupaviek:", bold_prefix="")
    add_bullet_item(doc, "Celková tlaková sila v kĺboch (F_joint ≈ T_FDP + T_FDS + F_ext) rastie s dĺžkou pák a telesnou hmotnosťou. Po rokoch lezenia to u starších lezcov zvyšuje náchylnosť na osteoartrózu PIP a DIP kĺbov, pričom prirodzená obnova kolagénu s vekom postupne spomaľuje.")
    add_body_paragraph(doc, "Odporúčania pre tréningovú prax:", bold_prefix="")
    add_bullet_item(doc, "Odporúča sa uprednostňovať otvorený úchop (open hand), úchop na troch prstoch (three-finger drag) a dlhšie, kontrolované submaximálne visy zamerané na adaptáciu šliach (60–70 % maxima na lištách 20–25 mm), vždy po dôkladnom 20 až 30-minútovom postupnom rozcvičení.", bold_prefix="Šetrenie kĺbov zmenšením pákového zaťaženia: ")

    # Table 2: Lifespan Summary
    add_heading2(doc, "3.5 Prehľad pákových pomerov a rizík podľa veku")
    
    t2_caption = doc.add_paragraph()
    t2_caption.paragraph_format.space_before = Pt(6)
    t2_caption.paragraph_format.space_after = Pt(3)
    t2_caption.paragraph_format.first_line_indent = 0
    r_cap2 = t2_caption.add_run("Tabuľka 2: Pákové pomery článkov prstov a rizikový profil podľa veku")
    format_run(r_cap2, size_pt=10.5, bold=True, italic=True)
    
    t2 = doc.add_table(rows=5, cols=6)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2, color="B0C4DE", sz="4")
    t2_widths = [Cm(2.0), Cm(2.6), Cm(2.4), Cm(2.8), Cm(3.1), Cm(3.1)]
    
    t2_headers = [
        "Veková kategória",
        "Vývoj dĺžky článkov (ΔL / Δt)",
        "Relatívna hĺbka na 15 mm lište (ρ = d / L_DP)",
        "Vnútorné sily (T_FDP, F_A2)",
        "Najčastejšie miesto poškodenia",
        "Odporúčania pre úchopy a tréning na hangboarde"
    ]
    t2_data = [
        (
            "Deti (do 12–13 r.)",
            "Plynulý rast (L_tot ≈ 50–65 mm)",
            "ρ > 1,0–1,2 (bežné lišty pôsobia ako hlboké chyty)",
            "Nízky absolútny moment, ale malé vnútorné rameno šľachy (h ≈ 3 mm)",
            "Neuzavretá rastová platnička (fýza) na báze stredného článku (epifyziolýza / PPSI)",
            "Vynechať izolovaný hangboard a mikro-lišty (d < 12 mm); dôraz na všestrannosť, madlá a obliny."
        ),
        (
            "Mládež a juniori (13–18 r.)",
            "Prudký rastový špurt v období PHV",
            "ρ ≈ 0,8–1,0 (prechodová oblasť)",
            "Rýchly nárast momentu, ktorý predbieha adaptáciu šliach a kostí",
            "Stresové poranenie rastovej platničky (PPSI 1–5); ruptúry pútok sú zriedkavé",
            "Visy s vlastnou hmotnosťou na hlbších lištách (≥ 20 mm); sledovať rastový špurt; obmedziť dynamický campus."
        ),
        (
            "Dospelí (18–45 r.)",
            "Ustálená dĺžka; rozdiely medzi morfotypmi (80–110 mm)",
            "ρ ∈ [0,79; 1,07] (výrazný vplyv hĺbky lišty)",
            "Páková nevýhoda +17,7 % pri rovnakej sile (ma ∝ L^0,50); +118,1 % nárast sily na pútko vplyvom hmotnosti",
            "Akútna ruptúra pútka A2 pri rázovom zaťažení (>400 N) a chronický zápal šľachovej pošvy",
            "Dávkovanie objemu podľa morfotypu; prispôsobenie testovacej lišty (0,80 L_DP); prevaha otvoreného úchopu."
        ),
        (
            "Masters (45+ r.)",
            "Ustálená dĺžka (80–110 mm)",
            "ρ ∈ [0,79; 1,07] (výrazný vplyv hĺbky lišty)",
            "Vysoké kumulatívne tlakové zaťaženie v kĺboch počas života",
            "Opotrebovanie kĺbovej chrupavky, osteoartróza PIP kĺbu, pomalšia regenerácia",
            "Odľahčenie kĺbov (otvorený úchop / obliny); submaximálne visy; dlhšie rozcvičenie a regenerácia."
        )
    ]
    
    hdr2_row = t2.rows[0]
    hdr2_trPr = hdr2_row._tr.get_or_add_trPr()
    hdr2_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    hdr2_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    for idx, text in enumerate(t2_headers):
        c = hdr2_row.cells[idx]
        c.width = t2_widths[idx]
        set_cell_background(c, "D9E2F3")
        set_cell_margins(c, top=80, bottom=80, left=80, right=80)
        p = c.paragraphs[0]
        p.paragraph_format.first_line_indent = 0
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        format_run(r, size_pt=9.5, bold=True, color=RGBColor(0x1F, 0x38, 0x64))
        
    for r_idx, row_data in enumerate(t2_data):
        row = t2.rows[r_idx + 1]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        bg_col = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, text in enumerate(row_data):
            c = row.cells[c_idx]
            c.width = t2_widths[c_idx]
            set_cell_background(c, bg_col)
            set_cell_margins(c, top=70, bottom=70, left=80, right=80)
            p = c.paragraphs[0]
            p.paragraph_format.first_line_indent = 0
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            format_run(r, size_pt=9.0, bold=(c_idx == 0))
            
    doc.add_paragraph() # Spacing

    # SECTION 4: Periodizácia podľa morfotypu
    add_heading1(doc, "4. Periodizácia podľa morfotypu prsta a návrh mikrocyklov [Biomechanická hypotéza a navrhovaný metodický rámec]")
    
    make_callout_box(
        doc,
        "Odborný prínos práce: Zohľadnenie morfotypu prsta v metodike tréningu",
        [
            "Systematické prehľady a metaanalýzy zamerané na tréning sily prstov v lezení (napr. Stien et al., 2023; Langer et al., 2023) aj kľúčové tréningové štúdie (López-Rivera & González-Badillo, 2012, 2019; Devise et al., 2022) potvrdzujú, že systematický tréning na hangboarde prináša výrazný nárast sily. Doterajšie výskumy však pracovali výlučne s vonkajšími parametrami tréningu – ako je veľkosť pridanej záťaže, intenzita, hĺbka lišty či trvanie intervalu práce a odpočinku – pričom k prstom všetkých lezcov pristupovali ako k rovnakej mechanickej páke. V odbornej literatúre doteraz chýbala metodika, ktorá by tréningové zaťaženie diferencovala podľa individuálnej dĺžky prstov, pomerov jednotlivých článkov a anatomického morfotypu.",
            "Hlavným cieľom tejto práce je práve prepojenie týchto biomechanických súvislostí s tréningovou praxou. Zohľadnením pomeru hĺbky chytu k dĺžke koncového článku (ρ = d / L_DP), sublineárneho rastu kĺbov (+17,7 % rozdiel v páke), nárastu telesnej hmotnosti s výškou postavy (+118,1 % nárast zaťaženia pútka) a opory viacerých článkov prsta ponúka navrhnutý model trénerom, inštruktorom a lezcom ucelený teoretický základ pre individualizáciu tréningu podľa morfotypu. Nasledujúce parametre týždenného mikrocyklu predstavujú metodický návrh a zároveň odbornú hypotézu pre budúce empirické štúdie v športovej biomechanike."
        ]
    )
    
    add_body_paragraph(
        doc,
        "Nasledujúca tabuľka porovnáva odporúčané nastavenie tréningu pre dva krajné morfotypy ruky:"
    )
    
    # Table 3: Periodization
    t3_caption = doc.add_paragraph()
    t3_caption.paragraph_format.space_before = Pt(6)
    t3_caption.paragraph_format.space_after = Pt(3)
    t3_caption.paragraph_format.first_line_indent = 0
    r_cap3 = t3_caption.add_run("Tabuľka 3: Odporúčané nastavenie tréningového mikrocyklu podľa morfotypu ruky")
    format_run(r_cap3, size_pt=10.5, bold=True, italic=True)
    
    t3 = doc.add_table(rows=7, cols=3)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3, color="B0C4DE", sz="4")
    t3_widths = [Cm(4.2), Cm(5.9), Cm(5.9)]
    
    t3_headers = ["Tréningový parameter", "Krátkoprstý morfotyp (L_tot < 85 mm)", "Dlhoprstý morfotyp (L_tot > 105 mm)"]
    t3_data = [
        ("Frekvencia max. visov na hangboarde", "2–3 tréningy týždenne", "najviac 1–2 tréningy týždenne"),
        ("Odpočinok medzi intenzívnym lištovaním", "24–48 hodín", "odporúčaných 48–72 hodín"),
        ("Odporúčaná testovacia lišta", "15 mm (d ≈ 0,80 L_DP)", "20–22 mm (d ≈ 0,80 L_DP)"),
        ("Pomer úchopov v tréningu", "50 % polouzamknutý úchop (half-crimp), 50 % otvorený úchop", "75 % otvorený úchop, 25 % polouzamknutý úchop (half-crimp)"),
        ("Prirodzene vyhovujúci charakter lezenia", "Previsy po malých lištách, technické steny", "Obliny, kompresné lezenie, stisky"),
        ("Dĺžka adaptačného mezocyklu pre šľachy a pútka", "6–8-týždňové tréningové bloky", "12–16-týždňové pozvoľné stupňovanie záťaže")
    ]
    
    hdr3_row = t3.rows[0]
    hdr3_trPr = hdr3_row._tr.get_or_add_trPr()
    hdr3_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    hdr3_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    for idx, text in enumerate(t3_headers):
        c = hdr3_row.cells[idx]
        c.width = t3_widths[idx]
        set_cell_background(c, "D9E2F3")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        p = c.paragraphs[0]
        p.paragraph_format.first_line_indent = 0
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        format_run(r, size_pt=10, bold=True, color=RGBColor(0x1F, 0x38, 0x64))
        
    for r_idx, row_data in enumerate(t3_data):
        row = t3.rows[r_idx + 1]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        bg_col = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, text in enumerate(row_data):
            c = row.cells[c_idx]
            c.width = t3_widths[c_idx]
            set_cell_background(c, bg_col)
            set_cell_margins(c, top=70, bottom=70, left=100, right=100)
            p = c.paragraphs[0]
            p.paragraph_format.first_line_indent = 0
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            format_run(r, size_pt=9.5, bold=(c_idx == 0))
            
    doc.add_paragraph() # Spacing

    # SECTION 5: Stručný fyzikálny prehľad
    add_heading1(doc, "5. Stručný fyzikálny prehľad: Teoretické východiská modelu")
    add_body_paragraph(
        doc,
        "Pre trénerov, inštruktorov a lezcov so záujmom o detailnejší pohľad na výpočtovú stránku práce sa navrhnutý 3D model opiera o štyri základné mechanické piliere:"
    )
    add_bullet_item(doc, "Kinematické transformačné matice opisujúce ohyb a vystretie (flexiu/extenziu) v MCP kĺbe (θ_MCP), bočný odklon (abdukciu/addukciu) v MCP kĺbe (φ_MCP), ohyb v PIP kĺbe (θ_PIP) a ohyb v DIP kĺbe (θ_DIP).", bold_prefix="1. Priestorová kinematika (reťazec so 4 stupňami voľnosti): ")
    add_bullet_item(doc, "Výpočet rozloženia kontaktného tlaku na koncovom článku (s ≤ L_DP) a strednom článku (s > L_DP). Pri opore stredného článku o chyt sa zapája pútko A3, čím vonkajší ohybový moment v DIP kĺbe klesá takmer na nulu.", bold_prefix="2. Kontaktná mechanika jedného a dvoch článkov: ")
    add_bullet_item(doc, "Normálová sila pôsobiaca na pútko vyplýva zo zmeny smeru šľachy pri prechode cez príslušné kĺby:", bold_prefix="3. Vychýlenie šľachy na pútku (kladkový / capstan model): ")
    
    add_equation_box(doc, "F_A2 = T_A2 · ( 0,50 (u_PP - u_MP) )")
    add_equation_box(doc, "F_A1 = T_A1 · ( u_wrist - u_PP )")
    add_equation_box(doc, "F_A4 = T_A4 · ( 0,40 (u_PP - u_MP) )")
    
    add_bullet_item(doc, "Bočný odklon prsta v MCP kĺbe (φ_MCP ≠ 0) premieta silu ťahu šľachy do čelnej (koronálnej) roviny, čím vznikajú bočné strihové sily (F_z) na vstupe do pútka A1 a na kolaterálnych väzoch PIP kĺbu.", bold_prefix="4. Strihové zaťaženie mimo roviny ohybu: ")

    # SECTION: Záver
    add_heading1(doc, "Záver: Dlhodobo udržateľné zdravie ako predpoklad výkonnosti")
    add_body_paragraph(
        doc,
        "V športovom lezení napreduje najrýchlejšie ten lezec, ktorý dokáže trénovať plynulo a dlhodobo bez nútených výpadkov spôsobených zraneniami. Adaptáciu šliach, pútok a kĺbových puzdier nemožno urýchliť – ich prestavba prebieha rádovo pomalšie než rast svalovej sily. Poznanie priestorovej biomechaniky prsta umožňuje trénerom a lezcom plánovať prípravu s rozvahou, chrániť pohybový aparát pred preťažením a posúvať vlastné lezecké limity počas mnohých rokov."
    )

    # SECTION: Literatúra
    add_heading1(doc, "Literatúra")
    
    references = [
        "Devise, M., Le Mansec, Y., Bideau, N., Couceiro, M., Seifert, L., 2022. Comparison of finger flexor strength and endurance in rock climbers: a systematic review. Sports Biomech., 1–25. https://doi.org/10.1080/14763141.2022.2136051",
        "Langer, K., Simon, C., Wiemeyer, J., 2023. Strength training in rock climbing: a systematic review. Front. Sports Act. Living 5, 1083437. https://doi.org/10.3389/fspor.2023.1083437",
        "Lin, G.T., Cooney, W.P., Amadio, P.C., An, K.N., 1990. Mechanical properties of human pulley and tendon. J. Hand Surg. Br. 15(4), 429–434. https://doi.org/10.1016/0266-7681(90)90085-f",
        "López-Rivera, E., González-Badillo, J.J., 2012. The effects of two different hangboard training programs on maximal strength and endurance in rock climbers. J. Sports Sci. 30(4), 407–415. https://doi.org/10.1080/02640414.2011.643640",
        "López-Rivera, E., González-Badillo, J.J., 2019. Comparison of the effects of three hangboard strength and endurance training programs on grip endurance in sport climbers. J. Sports Sci. Med. 18(3), 483–495. https://pubmed.ncbi.nlm.nih.gov/31427871/",
        "Schöffl, I., Oppelt, K., Jüngert, J., Schweizer, A., Bayer, T., Neuhuber, W., Schöffl, V., 2009. The influence of concentric and eccentric loading on the finger pulley system. J. Biomech. 42(13), 2124–2128. https://doi.org/10.1016/j.jbiomech.2009.05.034",
        "Schöffl, V., Lutter, C., Popp, D., 2023. The middle phalanx stress fracture in adolescent sport climbers: Primary Periphyseal Stress Injury (PPSI). Wilderness Environ. Med. 34(2), 198–205. https://doi.org/10.1016/j.wem.2023.01.006",
        "Schöffl, V., et al., 2025. Primary periphyseal stress injuries (PPSI) in youth climbers: updated five-grade classification and consensus diagnostic framework. Sports Orthop. Traumatol. 41(1), 12–21. https://doi.org/10.1016/j.orthtr.2024.11.002",
        "Stien, N., Saeterbakken, A.H., Hermans, E., Vereide, V.A., Andersen, V., 2023. Comparison of finger flexor strength training protocols in rock climbers: a systematic review. Sports Med. Open 9, 23. https://doi.org/10.1186/s40798-023-00569-w"
    ]
    
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.style = doc.styles['Normal']
        pf_ref = p_ref.paragraph_format
        pf_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf_ref.line_spacing = 1.25
        pf_ref.space_before = Pt(2)
        pf_ref.space_after = Pt(4)
        pf_ref.left_indent = Cm(0.8)
        pf_ref.first_line_indent = Cm(-0.8)
        
        # split url if present
        if "http" in ref:
            parts = ref.split("http")
            r_text = p_ref.add_run(parts[0])
            format_run(r_text, size_pt=10.5)
            r_url = p_ref.add_run("http" + parts[1])
            format_run(r_url, size_pt=10, italic=False, color=RGBColor(0x05, 0x63, 0xC1))
            r_url.font.underline = True
        else:
            r_text = p_ref.add_run(ref)
            format_run(r_text, size_pt=10.5)

    print(f"Saving final document to: {OUTPUT_PATH}")
    doc.save(OUTPUT_PATH)
    print("Document successfully created and saved!")

if __name__ == "__main__":
    build_document()

