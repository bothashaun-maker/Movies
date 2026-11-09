"""Build the printable guide and download bundle; never touches notebooks/models.
Requires reportlab==4.4.9 and system DejaVu Sans fonts.
"""
from pathlib import Path
import html,re,zipfile
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,KeepTogether

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
FONT=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('JJP','DejaVuSans.ttf'),('JJP-Bold','DejaVuSans-Bold.ttf'),('JJP-Italic','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('JJP',normal='JJP',bold='JJP-Bold',italic='JJP-Italic',boldItalic='JJP-Bold')
INK=colors.HexColor('#18212a');GOLD=colors.HexColor('#a57022');PALE=colors.HexColor('#f2f4f5')
styles=getSampleStyleSheet()
for name in ('BodyText','Normal','Heading1','Heading2','Heading3'):
    styles[name].fontName='JJP';styles[name].textColor=INK
styles['BodyText'].fontSize=9.3;styles['BodyText'].leading=14.2;styles['BodyText'].spaceAfter=8
styles['Heading1'].fontName='JJP-Bold';styles['Heading1'].fontSize=23;styles['Heading1'].leading=29;styles['Heading1'].spaceAfter=18
styles['Heading2'].fontName='JJP-Bold';styles['Heading2'].fontSize=15;styles['Heading2'].leading=20;styles['Heading2'].spaceBefore=14;styles['Heading2'].spaceAfter=10
styles['Heading3'].fontName='JJP-Bold';styles['Heading3'].fontSize=11;styles['Heading3'].leading=16
styles.add(ParagraphStyle('Cell',parent=styles['BodyText'],fontSize=8,leading=11,spaceAfter=0))
styles.add(ParagraphStyle('Cover',fontName='JJP-Bold',fontSize=35,leading=43,textColor=INK,spaceAfter=22))
styles.add(ParagraphStyle('Label',fontName='JJP-Bold',fontSize=10,leading=16,textColor=GOLD,spaceAfter=16))

def inline(text):
    # Resolve local Markdown links to the published repository for PDF readers.
    placeholders={}
    def link(match):
        label,target=match.groups()
        if not target.startswith(('http://','https://')):
            target='https://github.com/bothashaun-maker/Movies/blob/main/'+str((ROOT/target).resolve().relative_to(REPO))
        key=f'JJPURLTOKEN{len(placeholders)}'
        placeholders[key]=f'<link href="{html.escape(target,quote=True)}" color="#365f80">{html.escape(label)}</link>'
        return key
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',link,text)
    text=html.escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',text)
    text=re.sub(r'`([^`]+)`',r'<font color="#365f80">\1</font>',text)
    for key,value in placeholders.items():text=text.replace(key,value)
    return text

def footer(canvas,doc):
    w,h=A4;canvas.setStrokeColor(GOLD);canvas.line(44,40,w-44,40)
    canvas.setFont('JJP',7);canvas.setFillColor(INK)
    canvas.drawString(44,27,'JESUSJUICEPRODUCTIONS · MOVIE CREATION · EDITION I')
    canvas.drawRightString(w-44,27,str(doc.page))

def build():
    story=[Spacer(1,70),Paragraph('JESUSJUICEPRODUCTIONS',styles['Label']),
           Paragraph('THE MOVIE<br/>CREATION<br/>LAUNCH SYSTEM',styles['Cover']),
           Paragraph('One song. One cast. A complete path to the finished film.',styles['Heading2']),
           Paragraph('References → short shots → edit → upscale → deliver → archive',styles['BodyText']),
           Spacer(1,35),Paragraph('EXISTING NOTEBOOKS · EIGHT EPISODE WORLDS · ONE OPERATOR',styles['Label']),
           Paragraph('A practical companion to your JJP Launch System. Includes H3 Next, the Current Working workflow, all five Pixa workflows, Director, Three-Step and Movie Upscaler.',styles['BodyText']),
           Spacer(1,20),Paragraph('Edition I · 9 November 2026<br/>Documentation and templates only. No notebook or model changes.',styles['BodyText']),PageBreak()]
    lines=(ROOT/'JJP_MOVIE_LAUNCH_SYSTEM.md').read_text().splitlines()
    # Skip duplicate document title; keep introduction and attribution.
    i=3
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('## '):
            story.append(PageBreak());story.append(Paragraph(inline(line[3:]),styles['Heading1']));i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]),styles['Heading2']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                parts=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',x) for x in parts):
                    rows.append([Paragraph(inline(x),styles['Cell']) for x in parts])
                i+=1
            widths=[145,170,192] if len(rows[0])==3 else [507/len(rows[0])]*len(rows[0])
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5d7bd')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),1,GOLD)]))
            story.extend([table,Spacer(1,12)]);continue
        text=line;i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||- |\d+\. )',lines[i]):
            text+=' '+lines[i].strip();i+=1
        if text.startswith('- '):text='• '+text[2:]
        story.append(Paragraph(inline(text),styles['BodyText']))
    doc=SimpleDocTemplate(str(ROOT/'JJP_MOVIE_LAUNCH_SYSTEM.pdf'),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=46,bottomMargin=55,
                          title='JesusJuiceProductions — Movie Creation Launch System',author='JesusJuiceProductions')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    with zipfile.ZipFile(ROOT/'JJP_MOVIE_LAUNCH_KIT.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in [ROOT/'JJP_MOVIE_LAUNCH_SYSTEM.pdf',ROOT/'JJP_MOVIE_LAUNCH_SYSTEM.md',ROOT/'README.md',*sorted((ROOT/'templates').glob('*'))]:
            archive.write(path,path.relative_to(ROOT))
    print('Built PDF and launch-kit ZIP')

if __name__=='__main__':build()
