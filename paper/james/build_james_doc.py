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
        rPr_p = OxmlElement('w:rPr')
        b = OxmlElement('w:b')
        bCs = OxmlElement('w:bCs')
        rPr_p.append(b)
        rPr_p.append(bCs)
        pPr.append(rPr_p)
    elif level == 2:
        ind = OxmlElement('w:ind')
        ind.set(qn('w:left'), '216')
        pPr.append(ind)
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
    ptab = OxmlElement('w:ptab')
    ptab.set(qn('w:relativeTo'), 'margin')
    ptab.set(qn('w:alignment'), 'right')
    ptab.set(qn('w:leader'), 'dot')
    r2.append(ptab)
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
    borders = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
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
    set_cell_margins(cell, top=140, bottom=140, left=240, right=180)
    
    # Left border thick, others none
    set_cell_borders(cell, left={'val': 'single', 'sz': 24, 'color': border_color})
    
    # Add title
    p0 = cell.paragraphs[0]
    p0.style = doc.styles['Normal']
    p0.paragraph_format.first_line_indent = 0
    p0.paragraph_format.line_spacing = 1.2
    p0.paragraph_format.space_before = Pt(2)
    p0.paragraph_format.space_after = Pt(4)
    run_title = p0.add_run(title)
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(11)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    
    for text in paragraphs:
        p = cell.add_paragraph()
        p.style = doc.styles['Normal']
        p.paragraph_format.first_line_indent = 0
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10.5)
        
    doc.add_paragraph() # slight spacing after table

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

def add_heading1(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles['Heading 1']
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = 0
    pf.space_before = Pt(16)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    
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
    pf.first_line_indent = 0
    pf.space_before = Pt(12)
    pf.space_after = Pt(4)
    pf.keep_with_next = True
    
    p.text = ""
    r = p.add_run(text)
    format_run(r, font_name="Times New Roman", size_pt=13, bold=True, color=RGBColor(0x2F, 0x55, 0x97))
    return p

def add_heading3(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles['Heading 3']
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = 0
    pf.space_before = Pt(8)
    pf.space_after = Pt(2)
    pf.keep_with_next = True
    
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
        
    # Author at P20 (Cerovský bez mäkčeňa)
    doc.paragraphs[20].text = "Igor Cerovský"
    doc.paragraphs[20].alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in doc.paragraphs[20].runs:
        format_run(r, size_pt=16, bold=True)
        
    # Populate Table of Contents (sdt)
    sdt = doc._body._element.find(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sdt")
    content = sdt.find(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sdtContent")
    
    # Clear placeholder dummy entries from template
    for child in list(content):
        content.remove(child)
        
    content.append(make_toc_header())
    content.append(make_empty_p())
    
    toc_items = [
        ("Úvod", 3, 1),
        ("1. Základné biomechanické odporúčania pre tréning prstov", 3, 1),
        ("1.1 Rozpočtovanie úchopového zaťaženia – „pútka-chrániaci“ otvorený úchop", 3, 2),
        ("1.2 Kolinearita predlaktia a chytu & optimalizácia ťahového vektora", 4, 2),
        ("1.3 Periodizačný protokol pre dlhé prsty", 5, 2),
        ("1.4 Práca nôh ako „brnenie pre pútka“ – tlmenie dynamických rázov", 5, 2),
        ("1.5 Asymetria medzi prstami a ergonomický výber tréningových líšt", 6, 2),
        ("1.6 Ochrana kĺbového puzdra a kolaterálnych väzov", 6, 2),
        ("1.7 Normalizácia hĺbky lišty pri športovom testovaní", 7, 2),
        ("1.8 Tribológia kože, teplota a manažment trenia", 8, 2),
        ("2. Diagnostické sebahodnotenie: Preťažujete svoje pútka a kĺbové puzdrá?", 8, 1),
        ("3. Biomechanické usmernenia podľa vekových kategórií: Škálovanie pák článkov prstov a spojivových tkanív počas života", 9, 1),
        ("3.0 Princíp celoživotného škálovania pák", 9, 2),
        ("3.1 Deti (pediatrická kategória: <12–13 rokov / prepubertálne obdobie)", 9, 2),
        ("3.2 Juniori a dospievajúci (pubertálna kategória: ~13–18 rokov / rastový špurt)", 10, 2),
        ("3.3 Dospelí a hlavná kategória (~18–45 rokov / kostrová zrelosť)", 11, 2),
        ("3.4 Veteráni a masters (starší dospelí: 45–60+ rokov / pomalšia obnova tkanív)", 11, 2),
        ("3.5 Zhrnutie: Celoživotné škálovanie pák článkov prstov a biomechanický rizikový profil", 12, 2),
        ("4. Periodizácia zohľadňujúca fenotyp a návrh mikrocyklov", 12, 1),
        ("5. Stručný fyzikálny úvod: Veda v pozadí odporúčaní", 13, 1),
        ("Záver: Dlhovekosť predchádza výkonu", 14, 1),
        ("Literatúra", 14, 1)
    ]
    
    for title, page, level in toc_items:
        content.append(make_toc_entry(title, page, level=level))
        
    # Enable automatic field updates on document open in Word
    update_fields = OxmlElement('w:updateFields')
    update_fields.set(qn('w:val'), 'true')
    doc.settings.element.append(update_fields)
    
    # Manage page breaks and remove template instructions
    body = doc._body._element
    sdt_idx = None
    for idx, el in enumerate(body):
        if el.tag.endswith('sdt'):
            sdt_idx = idx
            break
            
    # Page break between Title page (P20) and Table of Contents (sdt)
    ns_w = nsdecls('w')
    br_before = parse_xml('<w:p ' + ns_w + '><w:r><w:br w:type="page"/></w:r></w:p>')
    body.insert(sdt_idx, br_before)
    
    # Page break after Table of Contents (sdt)
    br_after = parse_xml('<w:p ' + ns_w + '><w:r><w:br w:type="page"/></w:r></w:p>')
    body.insert(sdt_idx + 2, br_after)
    
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
    
    # --- PAGE 2 ONWARDS: BODY CONTENT ---
    
    # Metadata info card at the top of Page 2
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_widths = [Cm(4.5), Cm(11.5)]
    meta_data = [
        ("Autor:", "Igor Cerovský (Nezávislý výskumník v oblasti fyziky a biomechaniky lezenia)"),
        ("Dátum publikácie:", "14. október 2026"),
        ("Korešpondencia:", "igor.cerovsky@gmail.com"),
        ("Projektový repozitár:", "https://github.com/igorcerovsky/finger"),
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
        "Klasifikácia úrovne evidencie (Evidence-Tier):",
        [
            "V záujme vedeckej transparentnosti je každé odporúčanie klasifikované podľa primárnej úrovne podkladových dôkazov:",
            "• [Odvodené z modelu]: Kvantitatívne výsledky generované priamo naším simulačným výpočtovým aparátom pre 3D muskuloskeletálnu a kontaktnú mechaniku.",
            "• [Doložené literatúrou]: Zavedené empirické zistenia z klinickej športovej medicíny, kadaveróznych testov alebo tribologickej literatúry.",
            "• [Mechanistická hypotéza]: Logické dedukcie alebo trénerské heuristiky založené na mechanických princípoch, ktoré si vyžadujú ďalšiu priamu empirickú validáciu."
        ]
    )
    
    # Recommendation 1.1
    add_heading2(doc, "1.1 Rozpočtovanie úchopového zaťaženia – „Pútka-chrániaci“ otvorený úchop [Odvodené z modelu a doložené literatúrou]")
    add_body_paragraph(
        doc,
        "Pri uzavretom úchope (Full Crimp) núti hyperextenzia v DIP kĺbe šľachu hlbokého ohýbača (FDP) prenášať až 65 % celkového zaťaženia prsta, zatiaľ čo extrémny uhol flexie v PIP kĺbe (105°) smeruje vektor vychýlenia šľachy priamo proti pútku A2. Tým sa normálová sila na pútko A2 zvyšuje na 267,0 N pri štandardnom vise so záťažou 100 N na 10 mm lište (a na 457,5 N pri dynamickom zachytení so záťažou 171,7 N). Naopak, prechod na otvorený úchop (Open Hand) znižuje normálovú silu na pútko A2 na 62,6 N pri vise so záťažou 100 N (a na 107,5 N pri dynamickom zaťažení) – čo predstavuje viac ako 76 % zníženie namáhania pútka (pričom záťaž zostáva stabilne výrazne pod hranicou 100 N v celom rozsahu bežných tréningových záťaží).",
        bold_prefix="Fyzika úchopu: "
    )
    add_body_paragraph(doc, "Praktický tréningový protokol:", bold_prefix="")
    add_bullet_item(doc, "Približne 80 % celkového tréningového objemu (objemové lezenie, vytrvalostné intervaly, základné bloky na hangboarde) sa odporúča smerovať do otvoreného úchopu (Open Hand) a uvoľnenej polovičnej lišty (Half-Crimp).", bold_prefix="Pomer úchopov 80/20: ")
    add_bullet_item(doc, "Odporúča sa vnímať uzavretý úchop (full crimp so zámkom palca) ako vysoko intenzívnu rezervnú kapacitu. Je vhodné si ho vyhradiť primárne na kľúčové pokusy v projektoch na lištách pod 8 mm, kde ohybový moment v DIP kĺbe nie je možné vyvinúť alternatívnym úchopom. Odporúča sa vyhnúť trénovaniu uzavretých líšt až do úplného svalového vyčerpania.", bold_prefix="Racionálne dávkovanie uzavretej lišty: ")

    # Recommendation 1.2
    add_heading2(doc, "1.2 Kolinearita predlaktia a chytu & optimalizácia ťahového vektora – eliminácia priečneho laterálneho strihu [Odvodené z modelu a technická heuristika]")
    add_body_paragraph(
        doc,
        "Anulárne pútka (A2, A4) predstavujú anizotropné väzivovo-chrupavkové oblúky prispôsobené na obvodové napätie (udržiavanie šliach pri kosti). V lokálnom anatomickom súradnicovom systéme proximálneho článku vykazuje stred pútka A2 pri čistej sagitálnej flexii minimálne ohybové zaťaženie mimo roviny (FA2,lat_local ≈ 0 N). Keď sa však MCP kĺb dostane do radiálnej abdukcie mimo sagitálnej roviny (15°), vzniká výrazný priečny presmerovací strih koncentrovaný na pútku A1 / vstupnom palmárnom ústí pútka (137,3 N pri zaťažení 171,7 N; 80,4 N pri zaťažení 100 N). To je sprevádzané 44,4 N mediolaterálneho strihu na kolaterálnych väzoch PIP kĺbu (25,9 N pri 100 N zaťažení) a globálnou laterálnou zložkou 93,0 N (54,5 N pri 100 N zaťažení) odtiahnutia šľachy od kosti (bowstringing) na pútku A2.",
        bold_prefix="Fyzika biomechaniky: "
    )
    add_body_paragraph(doc, "Lezecká technika – ťažisko tela (CoG) a optimálny „ťahový vektor“:", bold_prefix="")
    add_bullet_item(doc, "Efektivita lezeckej techniky zásadne závisí od polohy tela. Aktívnym riadením ťažiska (boky, trup a dynamická práca nôh) môžu lezci smerovať výsledný ťahový vektor čo najkolmejšie na využiteľný povrch chytu.", bold_prefix="Ťažisko určuje nároky na prsty: ")
    add_bullet_item(doc, "Optimalizácia ťahového vektora prostredníctvom práce tela znižuje celkovú silu zovretia prstov potrebnú na udržanie kroku (maximalizuje efektívne trenie a mechanickú páku), vďaka čomu ťažké kroky pôsobia kontrolovateľnejšie. Súčasne minimalizuje parazitné krútiace momenty mimo roviny, priečny strih na vstupe do pútka A1 a namáhanie kolaterálnych väzov, čím znižuje dôležitý mechanický spúšťač akútnych zranení tkanív. Vybrúsená technika tak slúži ako účinná ochrana pred zranením.", bold_prefix="Dvojitý prínos (výkon + prevencia): ")
    add_bullet_item(doc, "Odporúča sa vyhnúť prepadávaniu lakťa dovnútra pri bočných ťahoch (gastonoch) alebo jeho neprimeranému vytáčaniu von („kuracie krídla“) pri zaťažení bočných líšt. Takéto zalomenie rotuje šľachovú pošvu a vytvára zvýšenú koncentráciu strihového napätia na bočných okrajoch pútok.", bold_prefix="Pasca polohy lakťov: ")
    add_body_paragraph(doc, "Bezpečnosť tréningu na hangboarde a so silomermi (Tindeq):", bold_prefix="")
    add_bullet_item(doc, "Pri testovaní alebo tréningu s izometrickými senzormi sily (napr. Tindeq Progressor, závesné váhy či inteligentné dosky) môžu lezci podvedome vytáčať zápästie, rozťahovať lakte alebo krútiť prsty, aby dosiahli vyššiu špičkovú nameranú hodnotu. Tým však vzniká nežiaduci kompenzačný vzorec: páčenie kĺbov a zapájanie svalov ramena a chrbta na úkor asymetrického strihu na vstupe do pútka A1 a preťaženia kolaterálnych väzov.", bold_prefix="Pasca honby za „maximálnym číslom“: ")
    add_bullet_item(doc, "Na hangboardoch alebo prenosných lištách sa odporúča polohovať prsty kolmo na hranu, aby sa minimalizovalo krútenie. Namiesto maximalizácie nominálneho čísla na senzore je vhodné uprednostniť čistú sagitálnu mechaniku, tréningovú efektivitu a dlhodobé zdravie kĺbov.", bold_prefix="Trénovať pre efektívnosť, nie pre falošné rekordy: ")

    # Recommendation 1.3
    add_heading2(doc, "1.3 Periodizačný protokol pre dlhé prsty [Odvodené z modelu a periodizačná hypotéza]")
    add_body_paragraph(
        doc,
        "Pri realistickom sublineárnom alometrickom škálovaní kondylov (ma ∝ L^0,50) majú lezci s dlhými prstami prirodzenú mechanickú nevýhodu páky +17,7 % na mikro-lištách (297,3 N oproti 252,6 N na 6 mm lište). Pri lezení s vlastnou hmotnosťou sa to kombinuje s alometrickým škálovaním telesnej hmotnosti (m ∝ L^2 až L^3), čo zvyšuje zaťaženie pútka A2 na 10 mm lište zo 174,2 N (krátke prsty) na 380,0 N (dlhé prsty) – čo predstavuje +118,1 % nárast zaťaženia, približujúci sa k hraniciam ruptúry zisteným na kadaveróznych vzorkách (~400 N). Navyše, plytké lišty poskytujú dlhým článkom menšie relatívne odľahčenie ohybového momentu (ρ = d / L_DP) a umiestnenie dlhších článkov na malú lištu si vyžaduje ostrejšie uhly flexie v PIP kĺbe (θ_PIP > 100°), čo ďalej zosilňuje odtiahnutie šľachy na pútku A2.",
        bold_prefix="Fyzika pákového pomeru: "
    )
    add_body_paragraph(doc, "Praktický tréningový protokol pre lezcov s dlhými prstami:", bold_prefix="")
    add_bullet_item(doc, "Kým kostrové svalstvo sa adaptuje na tréningové podnety v priebehu týždňov, syntéza kolagénu, tvorba priečnych väzieb a prestavba šľachových pútok prebiehajú výrazne pomalším metabolickým tempom (v praxi sa často odhaduje horizont 12 až 24 mesiacov postupného preťažovania). Lezcom s dlhšími prstami sa odporúča postupovať pri zmenšovaní chytov a pridávaní záťaže konzervatívnejšie ako ich rovesníkom s kratšími prstami.", bold_prefix="Progresívna periodizácia spojivových tkanív: ")
    add_bullet_item(doc, "Keďže dlhšie prsty zaznamenávajú vyššie absolútne napätie v tkanivách a vyšší tlak na kĺbové puzdrá pri lezení s vlastnou hmotnosťou, poskytnutie 48 až 72 hodín na regeneráciu spojivových tkanív medzi vysoko intenzívnymi tréningami prstov sa javí ako rozumné.", bold_prefix="Predĺžené regeneračné okná: ")
    add_bullet_item(doc, "Športovci s dlhými prstami prirodzene vynikajú na oblinách, veľkých stiskoch, otvorených štruktúrach a technických kompresných bouldroch, kde im rozpätie a plocha trenia poskytujú výraznú mechanickú výhodu. Naopak, na mikro-lištách čelia mechanickej nevýhode; namiesto silenia uzavretých líšt sa odporúča sústrediť na precíznu prácu nôh (vysoké stupy, vyvesenie do nôh, drop-knee) a otvorený ťahový úchop (drag).", bold_prefix="Využitie biomechanických predností: ")

    # Recommendation 1.4
    add_heading2(doc, "1.4 Práca nôh ako „brnenie pre pútka“ – tlmenie dynamických rázových preťažení [Doložené literatúrou a biomechanická extrapolácia]")
    add_body_paragraph(
        doc,
        "V statickej rovnováhe na 10 mm lište zaznamenávajú štandardné prsty silu na pútku A2 v rozsahu 265,1 až 267,0 N pri zaťažení prostredníka 100 N, čo sa nachádza vo fyziologických medziach. Keď sa však noha nečakane pošmykne alebo vyletí zo stupu, vonkajšie zaťaženie okamžite vyskočí na 25–40 % telesnej hmotnosti na jeden prst (171,7 N). Pri tomto zaťažení narastá normálová sila na pútko A2 pri štandardnom fenotype na 457,5 N, čo podstatne prekračuje kadaveróznu referenčnú hodnotu medze pevnosti v ťahu ~400 N (Lin et al. 1990; Schöffl et al. 2009).",
        bold_prefix="Fyzika rázového šoku: "
    )
    add_body_paragraph(doc, "Praktické tréningové návyky:", bold_prefix="")
    add_bullet_item(doc, "Odporúča sa osvojiť si návyk okamžite uvoľniť zovretie ruky a odskočiť, ak sa nohy nečakane pošmyknú, namiesto zúfalého kŕčovitého držania chytu za každú cenu. Akútne ruptúry pútok IV. stupňa sa často vyskytujú práve počas náhleho excentrického rázového zaťaženia po strate stupu.", bold_prefix="Reflex „pustiť pri šmyku“: ")
    add_bullet_item(doc, "Špičková práca nôh a spevnenie telesného jadra (core) poskytujú funkčnú ochranu flexorových pútok. Nácvik vedomého a presného kladenia nôh na šmykľavé stupy pomáha predchádzať nečakaným pošmyknutiam pri maximálnom zaťažení rúk.", bold_prefix="Cvičenia precíznosti nôh: ")
    add_bullet_item(doc, "Odporúča sa udržiavať gumu lezečiek čistú. Prach, zvyšky magnézia a nečistoty znižujú koeficient trenia (μ), najmä na ošmýkanom vápenci alebo hladkých no-tex / dual-tex stupoch na umelých stenách. Pravidelné očistenie podrážok lezečiek pred kľúčovým pokusom pomáha zabrániť náhlemu pošmyknutiu nôh, ktoré nárazovo zaťažuje pútka flexorov.", bold_prefix="Starostlivosť o lezečky a ich čistota: ")

    # Recommendation 1.5
    add_heading2(doc, "1.5 Asymetria medzi prstami a ergonomický výber tréningových líšt [Klinická heuristika doložená literatúrou]")
    add_body_paragraph(
        doc,
        "Na rovnej vodorovnej lište je prostredník (III. prst) výrazne dlhší ako ukazovák (II. prst) a prstenník (IV. prst). Aby prsty sedeli v jednej línii na plochej lište, prostredník je nútený do extrémnej hyperflexie v PIP kĺbe (>105°), čo na jeho pútku A2 koncentruje neprimerané normálové sily a krútiace momenty. Navyše, pretože prsty III, IV a V zdieľajú spoločné svalové bruško hlbokého ohýbača (efekt Quadriga), nerovnomerná extenzia prstov vytvára vysoké intertendinózne strihové napätie v lumbrikálnych svaloch.",
        bold_prefix="Fyzika lumbrikálneho strihu: "
    )
    add_body_paragraph(doc, "Praktické odporúčania pre vybavenie a tréning:", bold_prefix="")
    add_bullet_item(doc, "Odporúča sa voliť posilňovacie dosky s mierne zakrivenými, odstupňovanými alebo zošikmenými lištami, ktoré rešpektujú prirodzený dĺžkový gradient prstov. To umožňuje rovnomernejšie rozloženie záťaže na všetky štyri prsty bez toho, aby bol prostredník nútený do preťažujúcej hyperflexie.", bold_prefix="Anatomicky tvarované lišty: ")
    add_bullet_item(doc, "Pri tréningu dvojprstových úchopov (prostredník–prstenník) sa odporúča vyhnúť sa tesnému zovretiu nepoužitého ukazováka a malíčka do dlane. Udržiavanie voľných prstov v uvoľnenej, mierne pootvorenej polohe pomáha chrániť lumbrikálne svaly pred natrhnutím.", bold_prefix="Bezpečnosť pri dvojprstových dierach: ")

    # Recommendation 1.6
    add_heading2(doc, "1.6 Ochrana kĺbového puzdra a kolaterálnych väzov – riziko pošmyknutia s „unikajúcim prostredníkom“ [Klinické pozorovanie a mechanistická hypotéza]")
    add_body_paragraph(doc, "Fyzika postupného zošmykovania prstov:", bold_prefix="")
    add_bullet_item(doc, "V anatómii ľudskej ruky prostredník (III. prst) prečnieva o 8 až 15 mm pred susedný ukazovák (II) a prstenník (IV).", bold_prefix="Dĺžková disproporcia prstov: ")
    add_bullet_item(doc, "Keď sa ruka nečakane začne šmýkať dozadu alebo vyletí z chytu (v dôsledku potu, vyleštenej skaly, mastných dual-tex chytov či prudkého dynamického pohybu), prsty sa neodpájajú synchrónne. Keďže prsty II a IV sú kratšie, skĺznu z hrany ako prvé.", bold_prefix="Kaskáda postupného pošmyknutia: ")
    add_bullet_item(doc, "Počas kritického intervalu 20–60 milisekúnd pri pošmyknutí sa 100 % hybnosti padajúceho tela a unikajúceho záberu náhle prenesie na jediný prečnievajúci prostredník.", bold_prefix="Koncentrácia impulzu na jediný prst: ")
    add_bullet_item(doc, "Ako je prostredník ťahaný cez ostrú hranu chytu pod zaťažením plnej hmotnosti tela, podstupuje prudkú, vysokorýchlostnú excentrickú extenziu. Navyše, keďže povrchy chytov a ťahové vektory sú zriedkavo čisto kolmé, unikajúci prst zažíva intenzívny torzný krútiaci moment (M = r × F_slip).", bold_prefix="Nútená dynamická excentrická extenzia a torzia: ")
    add_bullet_item(doc, "Lezci s dlhými prstami majú podstatne dlhšie páky stredného a distálneho článku (L_MP + L_DP). To priamo znásobuje vonkajší krútiaci moment pôsobiaci na PIP kĺb a vystavuje puzdrové tkanivá závažným špičkovým torzným momentom.", bold_prefix="Násobenie pákového ramena pri dlhých prstoch: ")
    
    add_body_paragraph(doc, "Patomechanika kĺbového puzdra a palmárnej doštičky:", bold_prefix="")
    add_bullet_item(doc, "Kým anulárne pútka zlyhávajú primárne pod vplyvom obvodového ťahu smerujúceho dopredu (bowstringing), väzivové puzdro PIP kĺbu, kolaterálne väzy a väzivovo-chrupavková palmárna (volárna) doštička nesú hlavnú záťaž dynamickej hyperextenzie a laterálno-rotačného strihu.")
    add_bullet_item(doc, "Táto traumatická trakcia spôsobuje mikroskopické trhlinky v úponoch kolaterálnych väzov a preťažuje úpon palmárnej doštičky, čo vedie k akútnej alebo chronickej kapsulitíde / synovitíde PIP kĺbu.")
    add_bullet_item(doc, "Na rozdiel od akútneho puknutia pútka sa poranenie kĺbového puzdra prejavuje ako hlboká, difúzna obvodová bolesť okolo PIP kĺbu („opuchnutý / hrubý kĺb“), citlivosť pri stlačení kĺbu zbokov, ranná stuhnutosť a ostrá bolesť pri náhodnom bočnom ťuknutí alebo vytočení prsta.", bold_prefix="Klinický obraz: ")

    add_body_paragraph(doc, "Praktická prevencia a klinický manažment:", bold_prefix="")
    add_bullet_item(doc, "Trénujte potlačenie reflexu zachytiť pád zúfalým zaťatím jediného unikajúceho prostredníka. Ak sa ruka začne nekontrolovateľne šmýkať, odporúča sa chyt vedome a čisto pustiť, než nechať prostredník ťahať celú hmotnosť tela cez hranu.", bold_prefix="Odstránenie reflexu „zúfalého záseku“: ")
    add_bullet_item(doc, "Lezcom s výrazným dĺžkovým rozdielom prostredníka alebo opakovaným podráždením kĺbového puzdra sa odporúča aplikovať podporné X-tejpovanie cez palmárnu stranu PIP kĺbu. Prekrížené pásky mechanicky kladú odpor nútenej hyperextenzii a bočnému vychýleniu bez obmedzenia bežnej flexie úchopu.", bold_prefix="Funkčné tejpovanie PIP kĺbu do kríža (X-tejpovanie): ")
    add_bullet_item(doc, "Pri boulderingu na šmykľavých alebo dynamických oblinách poskytuje voľné zatejpovanie prostredníka k prstenníku (prst III k prstu IV pružnou páskou medzi MCP a PIP kĺbom) kľúčové bočné spevnenie a rozkladá sily pri pošmyknutí na oba prsty.", bold_prefix="Fixácia susedných prstov („buddy-taping“) pri náročných tréningoch: ")
    add_bullet_item(doc, "Nútenie opuchnutého kĺbu so zapáleným puzdrom do pasívnej extenzie zhoršuje podráždenie synovie.", bold_prefix="Vyhnúť sa agresívnemu pasívnemu naťahovaniu: ", level=1)
    add_bullet_item(doc, "Odporúča sa vykonávať jemné, submaximálne izometrické cvičenia v polovičnej lište alebo otvorenom úchope pri flexii 45°–60°, ktoré podporujú cirkuláciu synoviálnej tekutiny a remodeláciu kolagénu.", bold_prefix="Aktívna izometria v strednom rozsahu: ", level=1)
    add_bullet_item(doc, "Striedavé ponáranie do teplej a chladnej vody pomáha redukovať pretrvávajúci pozáťažový kĺbový výpotok a urýchľuje látkovú výmenu v málo cievne zásobenom tkanive kĺbového puzdra.", bold_prefix="Kontrastná hydroterapia: ", level=1)

    # Recommendation 1.7
    add_heading2(doc, "1.7 Normalizácia hĺbky lišty pri športovom testovaní [Hypotéza odvodená z modelu]")
    add_body_paragraph(
        doc,
        "Štandardné testovacie protokoly hodnotia všetkých lezcov na jednotnej 20 mm lište. Biomechanicky sa však relatívna hĺbka lišty mení v pomere k dĺžke distálneho článku (d / L_DP):",
        bold_prefix="Mýtus univerzálnej 20 mm lišty: "
    )
    add_bullet_item(doc, "Pri výrazne krátkoprstom fenotype (spodné percentily, L_DP = 18,7 mm) predstavuje 20 mm hlboký chyt (d / L_DP = 1,07). Zapája sa stredný článok, zaťaženie sa prenáša na pútko A3 a nároky na šľachu FDP dramaticky klesajú.")
    add_bullet_item(doc, "Pri výrazne dlhoprstom fenotype (horné percentily, L_DP = 25,3 mm) predstavuje 20 mm plytkú mikro-lištu (d / L_DP = 0,79), kde sa celé zaťaženie sústredí na koncový článok, čo prudko zvyšuje nároky na ohýbače.")
    add_body_paragraph(doc, "Praktické odporúčanie pre testovanie:", bold_prefix="")
    add_body_paragraph(doc, "Odporúča sa štandardizovať testovacie lišty podľa individuálnej anatómie lezca podľa vzorca:")
    add_equation_box(doc, "d_test = 0,80 × L_DP")
    add_bullet_item(doc, "Pre priemerný štandardný prst (L_DP ≈ 22 mm) sa odporúča testovať na lište 17,5 až 18 mm.")
    add_bullet_item(doc, "V okrajových prípadoch: pre krátke prsty (18,7 mm) sa rozmer upravuje na 15 mm, kým pre extrémne dlhé prsty (25,3 mm) sa rozmer posúva na 20 až 22 mm.")

    # Recommendation 1.8
    add_heading2(doc, "1.8 Tribológia kože, teplota a manažment trenia [Doložené literatúrou a tribologická heuristika]")
    add_body_paragraph(
        doc,
        "Trenie medzi pokožkou a skalou má nelineárny charakter a pri vysokom zaťažení klesá (μ ∝ F_N^(n-1)). Epidermálna vrstva kože sa správa ako viskoelastický kompozit; trenie sleduje krivku tvaru obráteného U v závislosti od teploty a hydratácie stratum corneum:",
        bold_prefix="Fyzika trenia a tribológia: "
    )
    add_bullet_item(doc, "Keď teplota okolia stúpne (>20 °C) alebo sa ruky potia, vytvára sa tekutý mazací film, ktorý spôsobuje prepad koeficientu trenia μ z ≈ 0,65 pod hodnotu 0,35. Ako kompenzáciu lezec prirodzene stláča chyt vyššou normálovou silou, čo kaskádovito vedie k vysokému napätiu v šľachách.", bold_prefix="Teplý a spotený režim („uvarenie chytu / greasing out“): ")
    add_bullet_item(doc, "Naopak, extrémny chlad (<5 °C), veľmi nízka vlhkosť vzduchu (<30 % RH) alebo nadmerne premagneziovaná pokožka vyvolávajú opačný, rovnako nebezpečný jav známy v lezeckom slangu ako „dry-firing“. V mrazivom a suchom vzduchu stráca stratum corneum svoju viskoelastickú prispôsobivosť a mení sa na tvrdý, sklovitý povrch. Bez základnej vnútornej vlhkosti sa koža nedokáže mikroskopicky deformovať do nerovností skaly. Končeky prstov sa z chytu okamžite zošmyknú bez akéhokoľvek hmatového varovania, čo vysiela netlmené dynamické rázové vlny priamo do anulárnych pútok a kĺbových puzdier.", bold_prefix="Extrémne chladný a vysušený režim („vyšmyknutie na sucho / dry-firing“): ")
    add_body_paragraph(doc, "Praktický protokol starostlivosti o kožu:", bold_prefix="")
    add_bullet_item(doc, "Najvhodnejšie podmienky pre trenie nastávajú v chladnom, sviežom prostredí (≈ 8–15 °C s miernou relatívnou vlhkosťou 40–60 %), kedy je pokožka dostatočne tuhá, aby odolávala strihu, no zároveň dostatočne poddajná na mikroskopické uzamknutie v štruktúre materiálu.", bold_prefix="Optimálna zóna trenia: ")
    add_bullet_item(doc, "Počas mrazivých suchých dní sa odporúča vyhnúť sa nadmernému nabaľovaniu magnézia na už vysušené ruky. Prsty je vhodné zohriať vo vreckách, prípadne pred nástupom na končeky prstov jemne dýchnuť teplý vlhký dych. Toto minimálne množstvo vlhkosti obnoví elasticitu pokožky bez vyvolania tekutého potu.", bold_prefix="Prevencia vyšmyknutia na sucho: ")
    add_bullet_item(doc, "Pravidelné čistenie chytov kefkou odstraňuje vrstvy vylešteného nánosu a usadeného magnézia a obnovuje mikroštruktúru povrchu. Vhodná kombinácia tekutého kriedového základu s čistým uhličitanom horečnatým udržiava ruky suché bez toho, aby tkanivo vysušila do sklovitého, krehkého stavu.", bold_prefix="Hygiena čistenia chytov a magnézium: ")

    # SECTION 2: Diagnostické sebahodnotenie
    add_heading1(doc, "2. Diagnostické sebahodnotenie: Preťažujete svoje pútka a kĺbové puzdrá?")
    add_body_paragraph(
        doc,
        "Tento klinický kontrolný zoznam založený na dôkazoch slúži lezcom a trénerom na identifikáciu mikropoškodení ešte pred vznikom akútneho štrukturálneho zlyhania tkanív:"
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
    
    t1_headers = ["Diagnostický príznak", "Anatomická štruktúra", "Biomechanická príčina", "Odporúčané opatrenie"]
    t1_data = [
        (
            "Ranná stuhnutosť pri báze proximálneho článku",
            "Pošva pútka A2",
            "Opakované vysoké obvodové napätie pri chronickom lištovaní",
            "Znížiť objem lištovania o 50 %; na 10 dní prejsť na otvorený úchop."
        ),
        (
            "Ostrá bolesť pri tlaku na palmárnu bázu PIP kĺbu",
            "Distálne A2 / proximálne A3",
            "Odtrhové napätie (peel stress) z hyperextenzie v DIP kĺbe pri uzavretom úchope",
            "Odporúča sa okamžite prerušiť uzavretý úchop; podpora H-tejpovaním alebo termoplastickým prstencom."
        ),
        (
            "Bolesť pri ťahu na bočných chytoch / gastonoch",
            "Vstup pútka A1 / bočné okraje pútok / kolaterálne väzy",
            "Nadmerný strih pri radiálnej abdukcii v MCP kĺbe (FA1,lat > 80 N, FPIP,ML > 40 N)",
            "Skontrolovať líniu predlaktia; vyhnúť sa vytáčaniu ruky na chyte."
        ),
        (
            "Bolesť medzi prstenníkom a prostredníkom",
            "Strih v lumbrikálnom svale",
            "Efekt Quadriga pri zaťažení hlbokých dier so zovretými voľnými prstami v dlani",
            "Odporúča sa prerušiť tréning 2-prstových dier; trénovať výhradne štvorprstový otvorený úchop."
        ),
        (
            "Hlboká obvodová bolesť PIP kĺbu; bočná citlivosť po pošmyknutí ruky",
            "Kĺbové puzdro PIP, palmárna doštička a kolaterálne väzy",
            "Postupné zošmykovanie prstov pri dynamickom pošmyknutí; dlhý prostredník absorbuje 100 % hybnosti pádu pri prudkej excentrickej extenzii a torznom strihu.",
            "Zastaviť reflex zúfalého držania pri šmyku; aplikovať X-tejpovanie PIP kĺbu alebo fixáciu k susednému prstu (prsty III+IV); aktívna izometria v strednom rozsahu; vyhnúť sa pasívnej hyperextenzii kĺbu, kým synovitída neustúpi."
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
    add_heading1(doc, "3. Biomechanické usmernenia podľa vekových kategórií: Škálovanie pák článkov prstov a spojivových tkanív počas života")
    
    add_heading2(doc, "3.0 Princíp celoživotného škálovania pák")
    add_body_paragraph(
        doc,
        "Náš 3D biomechanický model poukazuje na to, že vnútorné napätie v šľachách ohýbačov (T_FDP, T_FDS), kĺbové reakčné sily (F_joint) a normálové zaťaženia anulárnych pútok (F_A2, F_A4) sú diktované momentovou rovnováhou okolo jednotlivých kĺbov prsta:"
    )
    add_equation_box(doc, "M_joint = r_ext(L_DP, L_MP, L_PP) × F_ext")
    add_equation_box(doc, "T_FDP ≈ M_PIP / h_FDP ,        F_pulley = 2 T_FDP · sin(θ / 2)")
    add_body_paragraph(
        doc,
        "Pretože vnútorné pákové rameno šľachy v osteofibróznom digitálnom kanáli rastie sublineárne s kalibrom skeletu (ma ∝ L^0,50), vnútorné napätie šliach a normálové sily na pútka rastú priamo s dĺžkou pák článkov prstov (L_DP, L_MP, L_PP)."
    )
    add_body_paragraph(
        doc,
        "Okrem toho redistribúcia kontaktu pozdĺž prsta závisí od bezrozmerného pomeru hĺbky chytu:"
    )
    add_equation_box(doc, "ρ = d / L_DP")
    add_body_paragraph(
        doc,
        "Keď je ρ ≤ 1,0, kontaktná sila je koncentrovaná ako jednočlánkový kontakt výlučne na distálnom článku. Keď je ρ > 1,0, stredný článok vytvára dvojčlánkový kontakt, čím prenáša časť normálovej sily na stredný článok, stabilizuje pútko A3 a znižuje vonkajší ohybový moment v DIP kĺbe k nule."
    )

    add_heading2(doc, "3.1 Deti (pediatrická kategória: <12–13 rokov / prepubertálne obdobie)")
    add_body_paragraph(doc, "Dĺžka pák článkov a škálovanie hĺbky chytu (d / L_DP):", bold_prefix="")
    add_bullet_item(doc, "Detské články prstov sú proporčne miniatúrne: L_DP ≈ 13–16 mm, s celkovou dĺžkou prsta L_tot ≈ 50–65 mm (v porovnaní s 22 mm a 95 mm u dospelých).")
    add_bullet_item(doc, "Na štandardných chytoch v lezeckých centrách pre dospelých (18–22 mm lišty) je pomer hĺbky chytu pre dieťa ρ = d / L_DP > 1,2–1,5. To, čo je pre dospelého „lišta“, funguje pre dieťa ako hlboká viacčlánková polica, kde sa stredný článok prirodzene opiera o chyt.", bold_prefix="Paradox obrátenia dospelých chytov: ")
    add_bullet_item(doc, "Naopak, ak stavitelia ciest alebo mládežnícke súťaže zaradia plytké chyty zmenšené na 6–10 mm, pomer klesá na ρ ≤ 0,6, čo koncentruje silu priamo na špičku distálneho článku.", bold_prefix="Riziko detských mikro-líšt: ")
    add_body_paragraph(doc, "Zraniteľnosť rastovej platničky voči pákam (PPSI / epifyziolýza):", bold_prefix="")
    add_bullet_item(doc, "Epifyzárna rastová platnička (fýza) na báze stredného článku pozostáva z neosifikovaných chrupavkových chondrocytov, ktoré predstavujú mechanicky najslabší článok detského skeletu.")
    add_bullet_item(doc, "Pri uzavretom úchope vytvára vysoká hyperextenzia v DIP a flexia v PIP kĺbe výrazné dorzálne strihové napätie priamo cez otvorenú rastovú štrbinu. Opakovaná mikrotrauma môže vyvolať chronické stresové poškodenie epifýzy na báze stredného článku (Primary Periphyseal Stress Injury – PPSI stupne 1–5; Schöffl et al., 2025).")
    add_body_paragraph(doc, "Praktické odporúčania podložené dôkazmi:", bold_prefix="")
    add_bullet_item(doc, "Pre prepubertálne deti sa neodporúčajú cielené protokoly na hangboarde, kampusové silové rebríky ani visy s pridanou záťažou, aby sa chránili otvorené rastové platničky.", bold_prefix="Dôrazné obmedzenie izolovaných visov na hangboarde a mikro-líšt: ")
    add_bullet_item(doc, "Odporúča sa preferovať ergonomické chyty (d ≥ 20 mm), obliny, štruktúry a kompresné tvary, kde sa stredný článok podieľa na kontakte (ρ > 1,0).", bold_prefix="Geometria chytov a podpora dvoch článkov (d ≥ L_DP): ")

    add_heading2(doc, "3.2 Juniori a dospievajúci (pubertálna kategória: ~13–18 rokov / rastový špurt)")
    add_body_paragraph(doc, "„Nárast pákových ramien“ počas vrcholu výškovej rýchlosti (PHV):", bold_prefix="")
    add_bullet_item(doc, "Počas obdobia najrýchlejšieho rastu (Peak Height Velocity – PHV) rýchly pozdĺžny rast článkov prstov predlžuje vonkajšie páky rýchlejšie, než stíha prebiehať remodelácia šliach a fýz.")
    add_bullet_item(doc, "Športovo-medicínsky výskum (Schöffl et al., 2023, 2025) ukazuje, že akútne ruptúry anulárnych pútok sú u adolescentov pomerne zriedkavé; primárnym miestom zlyhania býva otvorená fýza na báze stredného článku (klasifikovaná škálou PPSI stupne 1–5).", bold_prefix="Dominantná zraniteľnosť (PPSI): ")
    add_body_paragraph(doc, "Praktické odporúčania podložené dôkazmi:", bold_prefix="")
    add_bullet_item(doc, "Odporúča sa sledovať trendy telesnej výšky a rastu prstov na identifikáciu fázy PHV (>6–8 cm/rok) a počas rastových špurtov zmierniť objem lištovania.", bold_prefix="Sledovanie rýchlosti rastu: ")
    add_bullet_item(doc, "Vhodné je uprednostniť visy s vlastnou telesnou hmotnosťou na dostatočne hlbokých lištách (d ≥ 20–25 mm, ρ ≥ 1,0) v otvorenom úchope alebo uvoľnenej polovičnej lište; dynamické kampusové skoky je vhodné odložiť až po dosiahnutí kostrovej zrelosti.", bold_prefix="Konzervatívny tréning na hangboarde na podporných lištách (d ≥ 20 mm): ")

    add_heading2(doc, "3.3 Dospelí a hlavná kategória (~18–45 rokov / kostrová zrelosť)")
    add_body_paragraph(doc, "Fixná dĺžka článkov a fenotypové škálovanie (L_tot ∈ [80, 110 mm]):", bold_prefix="")
    add_bullet_item(doc, "Pri realistickom sublineárnom alometrickom škálovaní (ma ∝ L^0,50) majú dlhšie prsty prirodzenú +17,7 % mechanickú nevýhodu na mikro-lištách (297,3 N oproti 252,6 N na 6 mm lište pri záťaži 100 N).")
    add_bullet_item(doc, "Lišta s hĺbkou 20 mm pôsobí ako dvojčlánková polica pre dospelého s krátkymi prstami (L_DP = 18,7 mm, ρ = 1,07, zaťaženie A2 218,0 N), ale zostáva jednočlánkovou lištou pre dospelého s dlhými prstami (L_DP = 25,3 mm, ρ = 0,79, zaťaženie A2 263,6 N, štrukturálna penalizácia +20,9 %).", bold_prefix="Dvojčlánkový kontakt na štandardných testovacích lištách (20 mm): ")
    add_bullet_item(doc, "Pri podpore telesnej hmotnosti škálovanej k výške (m ∝ L^2) zaznamenávajú lezci s dlhými prstami (92,5 kg) zaťaženie pútka A2 na úrovni 380,0 N na 10 mm lište v porovnaní so 174,2 N u lezcov s krátkymi prstami (50,6 kg) – čo predstavuje +118,1 % nárast absolútneho zaťaženia, približujúci sa k limitom ruptúry pútka (~400 N).", bold_prefix="Alometrický nárast hmotnosti na malých lištách (10 mm): ")
    add_body_paragraph(doc, "Praktické odporúčania podložené dôkazmi:", bold_prefix="")
    add_bullet_item(doc, "Lezcom s dlhými prstami vyhovujú regeneračné okná 48–72 hodín medzi náročnými tréningami na lištách.", bold_prefix="Kalibrácia objemu a regenerácie podľa fenotypu: ")
    add_bullet_item(doc, "Nastavenie hĺbky testovacej lišty podľa individuálnej dĺžky distálneho článku poskytuje biomechanicky spravodlivé a porovnateľné hodnotenie.", bold_prefix="Anatomická normalizácia líšt (d_test ≈ 0,80 L_DP): ")

    add_heading2(doc, "3.4 Veteráni a masters (starší dospelí: 45–60+ rokov / pomalšia obnova tkanív)")
    add_body_paragraph(doc, "Kumulatívna celoživotná expozícia pák a kĺbové kontaktné napätie:", bold_prefix="")
    add_bullet_item(doc, "Kumulatívne kĺbové kontaktné napätie (F_joint ≈ T_FDP + T_FDS + F_ext) rastie s dĺžkou pák a telesnou hmotnosťou, čo predisponuje starších lezcov k osteoartróze kĺbov PIP a DIP a k pomalšiemu obratu kolagénu.")
    add_body_paragraph(doc, "Praktické odporúčania podložené dôkazmi:", bold_prefix="")
    add_bullet_item(doc, "Vhodné je preferovať otvorený úchop (open hand), ťah na troch prstoch (three-finger drag) a submaximálne kontrolované visy na hustotu spojiva (60–70 % MVC na lištách 20–25 mm) s dôkladným 20–30-minútovým progresívnym rozcvičením.", bold_prefix="Zmenšenie pák pre dlhovekosť kĺbov: ")

    # Table 2: Lifespan Summary
    add_heading2(doc, "3.5 Zhrnutie: Celoživotné škálovanie pák článkov prstov a biomechanický rizikový profil")
    
    t2_caption = doc.add_paragraph()
    t2_caption.paragraph_format.space_before = Pt(6)
    t2_caption.paragraph_format.space_after = Pt(3)
    t2_caption.paragraph_format.first_line_indent = 0
    r_cap2 = t2_caption.add_run("Tabuľka 2: Celoživotné škálovanie pák článkov prstov a biomechanický rizikový profil")
    format_run(r_cap2, size_pt=10.5, bold=True, italic=True)
    
    t2 = doc.add_table(rows=5, cols=6)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2, color="B0C4DE", sz="4")
    t2_widths = [Cm(2.0), Cm(2.6), Cm(2.4), Cm(2.8), Cm(3.1), Cm(3.1)]
    
    t2_headers = [
        "Veková kohorta",
        "Dynamika pák článkov",
        "Pomer hĺbky chytu (15 mm)",
        "Škálovanie vnútorných síl",
        "Primárny spôsob zlyhania",
        "Odporúčané zameranie úchopov"
    ]
    t2_data = [
        (
            "Deti (<12–13 r.)",
            "Progresívny lineárny rast (L_tot ≈ 50–65 mm)",
            "ρ > 1,0–1,2 (Hlboký chyt)",
            "Nízky absolútny moment, miniatúrne rameno šľachy",
            "Neosifikovaná fýza na báze stredného článku (Epifyziolýza / PPSI)",
            "Vyhnúť sa izolovanému hangboardu a mikro-lištám; dôraz na madlá, obliny."
        ),
        (
            "Juniori (13–18 r.)",
            "Rýchly rastový špurt počas PHV",
            "ρ ≈ 0,8–1,0 (Prechodová zóna)",
            "Prudký nárast momentu predstihujúci remodeláciu",
            "Primárne perifyzárne stresové poranenie (PPSI 1–5); ruptúry pútok zriedkavé",
            "Preferovať vlastnú váhu na hlbších lištách (≥ 20 mm); sledovať PHV."
        ),
        (
            "Dospelí (18–45 r.)",
            "Ustálená dĺžka; fenotypová variabilita (80–110 mm)",
            "ρ ∈ [0,79; 1,07] (Citlivé na hĺbku)",
            "Nevýhoda páky +17,7 %; +118,1 % nárast záťaže od hmotnosti",
            "Akútna ruptúra pútka A2 pri rázovom zaťažení (>400 N) a chronická tendovaginitída",
            "Objem rešpektujúci fenotyp; normalizácia lišty (0,80 L_DP); otvorený úchop."
        ),
        (
            "Masters (45+ r.)",
            "Ustálená dĺžka (80–110 mm)",
            "ρ ∈ [0,79; 1,07] (Citlivé na hĺbku)",
            "Vysoké celoživotné kumulatívne kontaktné napätie",
            "Úbytok kĺbovej chrupavky, osteoartróza PIP kĺbu, pomalšia regenerácia",
            "Zmiernenie pák (otvorený úchop / obliny); submaximálne visy; dlhší odpočinok."
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

    # SECTION 4: Periodizácia zohľadňujúca fenotyp
    add_heading1(doc, "4. Periodizácia zohľadňujúca fenotyp a návrh mikrocyklov [Biomechanická hypotéza a navrhovaný intervenčný rámec]")
    
    make_callout_box(
        doc,
        "Kľúčová výskumná hranica: Preklenutie medzery v tréningu podľa fenotypu",
        [
            "Systematické prehľady a meta-analýzy špecifického tréningu sily prstov v lezení (napr. Stien et al., 2023; Langer et al., 2023) a prelomové intervenčné štúdie (López-Rivera & González-Badillo, 2012, 2019; Devise et al., 2022) potvrdzujú, že hangboardové protokoly vyvolávajú významné adaptácie. Všetky doterajšie vedecké intervencie však manipulovali výlučne s externými premennými – ako je absolútna záťaž, percento intenzity, hĺbka lišty či intervaly zaťaženia a odpočinku – pričom ľudský prst jednotne chápali ako nemennú, rovnakú páku. Žiadna doterajšia kontrolovaná empirická štúdia nepredpísala tréning na základe individuálnej dĺžky prstov, proporcií článkov či anatomického fenotypu.",
            "Preklenutie tejto biomechanickej medzery je hlavným cieľom tejto práce. Zhodnotením 3D muskuloskeletálnej fyziky relatívneho kontaktu (ρ = d / L_DP), sublineárnej alometrie kondylov (+17,7 % rozdiel v páke), alometrického škálovania telesnej hmotnosti (+118,1 % nárast zaťaženia pútka) a kontaktnej mechaniky viacerých článkov poskytuje náš model trénerom, fyzioterapeutom a lezcom formalizovaný teoretický rámec pre individualizáciu tréningu prstov podľa fenotypu. Týždenné mikrocykly uvedené nižšie predstavujú štruktúrované, testovateľné hypotézy navrhnuté ako vodidlo pre budúce empirické intervenčné štúdie v športovej biomechanike."
        ]
    )
    
    add_body_paragraph(
        doc,
        "Nasledujúca tabuľka ilustruje hypotetické rozdiely v periodizácii podľa morfológie ruky:"
    )
    
    # Table 3: Periodization
    t3_caption = doc.add_paragraph()
    t3_caption.paragraph_format.space_before = Pt(6)
    t3_caption.paragraph_format.space_after = Pt(3)
    t3_caption.paragraph_format.first_line_indent = 0
    r_cap3 = t3_caption.add_run("Tabuľka 3: Hypotetické týždenné šablóny mikrocyklov podľa morfológie ruky")
    format_run(r_cap3, size_pt=10.5, bold=True, italic=True)
    
    t3 = doc.add_table(rows=7, cols=3)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3, color="B0C4DE", sz="4")
    t3_widths = [Cm(4.2), Cm(5.9), Cm(5.9)]
    
    t3_headers = ["Tréningová dimenzia", "Krátky fenotyp (L_tot < 85 mm)", "Dlhý fenotyp (L_tot > 105 mm)"]
    t3_data = [
        ("Frekvencia max. hangboardu", "2–3 tréningy / týždeň", "1–2 tréningy / týždeň max."),
        ("Odpočinok medzi lištovaním", "24–48 hodín", "48–72 hodín odporúčaných"),
        ("Preferovaná testovacia lišta", "15 mm (d ≈ 0,80 L_DP)", "20–22 mm (d ≈ 0,80 L_DP)"),
        ("Rozloženie úchopov", "50 % polovičná lišta, 50 % otvorený úchop", "75 % otvorený úchop, 25 % polovičná lišta"),
        ("Špecializácia na profily", "Lištové previsy, technické hrany", "Obliny, kompresie, stisky"),
        ("Makrocyklus spojivového tkaniva", "6–8-týždňové bloky zaťaženia", "12–16-týždňové pozvoľné nábehy")
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

    # SECTION 5: Stručný fyzikálny úvod
    add_heading1(doc, "5. Stručný fyzikálny úvod: Veda v pozadí odporúčaní")
    add_body_paragraph(
        doc,
        "Pre trénerov a lezcov, ktorých zaujímajú podkladové matematické formulácie, náš 3D model prepája štyri základné mechanické piliere:"
    )
    add_bullet_item(doc, "Kinematické transformačné matice mapujúce flexiu/extenziu v MCP kĺbe (θ_MCP), abdukciu/addukciu v MCP kĺbe (φ_MCP), flexiu v PIP kĺbe (θ_PIP) a flexiu v DIP kĺbe (θ_DIP).", bold_prefix="1. Priestorová kinematika (reťazec so 4 stupňami voľnosti): ")
    add_bullet_item(doc, "Rieši distribúciu kontaktného tlaku na distálnom článku (s ≤ L_DP) a strednom článku (s > L_DP). Zapojenie stredného článku aktivuje kostné ukotvenie pútka A3, čím posúva vonkajší moment pôsobiaci na DIP kĺb k nule.", bold_prefix="2. Kontaktná mechanika dvoch článkov: ")
    add_bullet_item(doc, "Sila presmerovania šľachy na pútku sa počíta z vektorového vychýlenia cez príslušné kĺby:", bold_prefix="3. Presmerovanie šľachy na pútku (Capstan model): ")
    
    add_equation_box(doc, "F_A2 = T_A2 · ( 0,50 (u_PP - u_MP) )")
    add_equation_box(doc, "F_A1 = T_A1 · ( u_wrist - u_PP )")
    add_equation_box(doc, "F_A4 = T_A4 · ( 0,40 (u_PP - u_MP) )")
    
    add_bullet_item(doc, "Nenulová radiálna abdukcia (φ_MCP ≠ 0) premieta vektor ťahu šľachy do koronálnej roviny, čo generuje bočné strihové sily (F_z) na vstupe do pútka A1 a na kolaterálnych väzoch PIP kĺbu.", bold_prefix="4. Strihové namáhanie mimo sagitálnej roviny: ")

    # SECTION: Záver
    add_heading1(doc, "Záver: Dlhovekosť predchádza výkonu")
    add_body_paragraph(
        doc,
        "V športovom lezení je najsilnejším lezcom ten, kto dokáže trénovať konzistentne a dlhodobo bez vynútených prestávok spôsobených zraneniami. Regeneráciu a adaptáciu šliach a pútok nemožno unáhliť; ich metabolický obrat je rádovo pomalší ako u svalového tkaniva. Porozumením trojrozmernej fyzike prsta môžu lezci trénovať s vedeckým zámerom, chrániť svoje anatomické štruktúry a liezť náročnejšie cesty počas mnohých rokov."
    )

    # SECTION: Literatúra
    add_heading1(doc, "Literatúra")
    
    references = [
        "Devise, M., Le Mansec, Y., Bideau, N., Bideau, N., Couceiro, M., Seifert, L., 2022. Comparison of finger flexor strength and endurance in rock climbers: a systematic review. Sports Biomech., 1–25. https://doi.org/10.1080/14763141.2022.2136051",
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
