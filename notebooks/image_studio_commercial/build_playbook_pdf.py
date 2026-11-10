"""Build the JJP image-production companion PDF. Requires ReportLab."""
from pathlib import Path
from xml.sax.saxutils import escape
import json, hashlib
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).parent
OUT=ROOT/'JJP_Image_Production_Playbook.pdf'
for name,file in [('JJP','DejaVuSans.ttf'),('JJP-Bold','DejaVuSans-Bold.ttf'),('JJP-Italic','DejaVuSans-Oblique.ttf')]:pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
pdfmetrics.registerFontFamily('JJP',normal='JJP',bold='JJP-Bold',italic='JJP-Italic',boldItalic='JJP-Bold')
W,H=540,765.12; M=43; CW=W-2*M
BG=HexColor('#101010');FG=HexColor('#eeeae6');MUTED=HexColor('#a49e9b');CORAL=HexColor('#df8588');CYAN=HexColor('#76dfe0');LINE=HexColor('#363130');PANEL=HexColor('#1d1718')
c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
c.setTitle('JesusJuiceProductions — The Image Production Playbook')
c.setAuthor('JesusJuiceProductions');c.setSubject('Commercial v7 image creation, face and head replacement, Kaggle model packs, upscaling and music-video reference workflow')
styles={}
for key,size,leading,color,font in [('body',9.7,14.2,FG,'JJP'),('small',8.4,12.5,MUTED,'JJP'),('h',14,19,CORAL,'JJP-Bold'),('call',11,16.5,FG,'JJP'),('tiny',7.4,11,MUTED,'JJP')]:styles[key]=ParagraphStyle(key,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=0)
page=0;y=0

def p(text,style='body',x=M,width=CW,gap=9):
 global y
 para=Paragraph(text,styles[style]);_,height=para.wrap(width,H)
 if y-height<52:raise ValueError(f'Overflow page {page}: {text[:60]} at {y}, height {height}')
 para.drawOn(c,x,y-height);y-=height+gap

def label(text,x=M,yy=None,color=CYAN,size=7.5):
 c.saveState()
 c.setFillColor(color);c.setFont('JJP-Bold',size);t=c.beginText(x,yy if yy is not None else y);t.setCharSpace(2.2);t.textLine(text.upper());c.drawText(t);c.restoreState()

def start(section,title,kicker):
 global page,y
 if page:c.showPage()
 page+=1;c.setFillColor(BG);c.rect(0,0,W,H,fill=1,stroke=0)
 if page>1:
  label('JesusJuiceProductions',yy=H-27,size=6,color=MUTED)
  c.setStrokeColor(LINE);c.line(M,H-37,W-M,H-37)
  c.setFont('JJP',7);c.setFillColor(MUTED);c.drawString(M,27,'IMAGE PRODUCTION · COMMERCIAL v7');c.drawRightString(W-M,27,f'{page-1:02d}')
  y=H-66;label('Section '+section);y-=33
  titlep=Paragraph(title,ParagraphStyle('title',fontName='JJP-Bold',fontSize=25,leading=29,textColor=FG));_,h=titlep.wrap(CW,H);titlep.drawOn(c,M,y-h);y-=h+14
  p(kicker,'small',gap=17)
  c.setStrokeColor(CORAL);c.setLineWidth(1);c.line(M,y,W-M,y);y-=23

def h(text):p(text,'h',gap=7)
def box(title,text):
 global y
 titlep=Paragraph(title,styles['h']);body=Paragraph(text,styles['body']);_,a=titlep.wrap(CW-28,H);_,b=body.wrap(CW-28,H);height=a+b+35
 if y-height<52:raise ValueError(f'Box overflow {page}: {title}')
 c.setFillColor(PANEL);c.setStrokeColor(LINE);c.roundRect(M,y-height,CW,height,4,fill=1,stroke=1)
 titlep.drawOn(c,M+14,y-13-a);body.drawOn(c,M+14,y-21-a-b);y-=height+12

def link(title,url):p(f'<link href="{escape(url)}" color="#76dfe0"><u>{escape(title)}</u></link>','small',gap=6)
def step(n,title,text):h(f'{n:02d}  {title}');p(text,gap=11)
BASE='https://github.com/bothashaun-maker/Movies'
DOC=BASE+'/blob/main/notebooks/image_studio_commercial/'
start('','', '')
# Cover follows the visual language of the supplied launch-system PDF.
c.setStrokeColor(CORAL);c.setLineWidth(.9)
for x,yy,dx,dy in [(18,H-18,14,-14),(W-18,H-18,-14,-14),(18,18,14,14),(W-18,18,-14,14)]:c.line(x,yy,x+dx,yy);c.line(x,yy,x,yy+dy)
c.setStrokeColor(HexColor('#426a6b'));c.roundRect(M,565,299,23,2,fill=0,stroke=1);label('JesusJuiceProductions',x=M+11,yy=573,size=7.3)
c.setFont('JJP-Bold',37);c.setFillColor(FG);c.drawString(M,512,'THE IMAGE')
c.drawString(M,466,'PRODUCTION');c.setFillColor(CORAL);c.drawString(M,420,'PLAYBOOK')
y=387;p('A practical companion to <b>The Launch System</b> — for cinematic music-video stills, a consistent anthology cast, face and head replacement, upscaling and finished delivery.','call',gap=25)
box('One cast. Every world.','Build the shot around the song. Keep the human leads recognisable, the visual world intentional and the Word central.')
y-=5;label('Create  /  Edit  /  Swap  /  Upscale',size=7.1);y-=28
p('COMMERCIAL v7 PROFILE<br/>Updated 10 November 2026 · South Africa','small')
c.setStrokeColor(LINE);c.line(M,90,W-M,90);label('Edition II · Image operations',yy=69,size=6.8,color=MUTED)
c.setFont('JJP',7);c.setFillColor(MUTED);c.drawRightString(W-M,49,'STILLS FIRST · CLIPS NEXT · MOVIE UPSCALE LAST')

start('00','Start with what you have','Your actual workflow, and the route through this companion.')
box('Your finished references are still useful','You confirmed that the face swaps were made in <b>Google Flow</b>, followed by the previous notebook’s <b>Upscale 4x</b> workflow. That upscale used Real-ESRGAN x4plus. The old notebook’s InsightFace restrictions do not apply to swaps you made in Flow.')
p('There is no reason to redo these images because of the ×4 upscaler. Google Flow work remains subject to Google’s applicable terms and your source-image and likeness permissions. Keep the originals alongside the enlarged masters.')
h('The production chain')
p('<b>Approved still → clip generation → edit the music video → movie upscale → final quality check.</b> A reference around 1024 pixels per side becomes roughly 4096 pixels per side after ×4. The video model may resize that reference internally; the movie upscaler handles the finished clip separately.')
h('Use this book in order')
for text in ['01 / Set up the two reusable Kaggle model packs.','02 / Create and edit images; establish the approved cast.','03 / Choose dedicated head replacement or a smaller face edit.','04 / Upscale once, inspect, and preserve the master.','05–06 / Match the image world to the episode.','07–08 / Export, archive and check the model record.']:p(text,'small',gap=5)
p('This book adapts the supplied JJP launch plan. It covers still-image production; it does not add video face tracking or promise temporal consistency.','small',gap=0)

start('01','Prepare once. Reuse.','Kaggle storage is the bridge between sessions, not another download to your PC.')
step(1,'Build the CPU model packs','Import <b>Image_Studio_Commercial_Model_Packs.ipynb</b>. Use CPU with Internet. Run with <b>PACK = \'create_edit\'</b> for the creation/editing weights (~16.13 GB). Save that output or Dataset. In a separate copy/run, use <b>PACK = \'finish\'</b> for the upscalers, landmarks and BFS adapter (~4.40 GB).')
step(2,'Let the notebook save privately','Automatic private Dataset submission is enabled. It uses your configured Kaggle authentication, a KAGGLE_API_TOKEN secret, or legacy KAGGLE_USERNAME + KAGGLE_KEY secrets. Set DATASET_OWNER if the token does not expose your username. Never put credentials into prompts or shared files.')
step(3,'Attach both packs to the studio','Import a fresh <b>Image_Studio_Commercial_v7.ipynb</b>. Add the two private Datasets or saved notebook outputs under Add Input. Select a GPU, enable Internet and run the single startup cell. Wait for the Cloudflare link, open ComfyUI, choose a workflow and press Run.')
box('Second run: reuse verified files','Matching filenames, sizes and SHA256 checks allow the saved weights to be reused. The studio does not redownload attached, verified models. An old pack can supply matching Real-ESRGAN/SeedVR2 weights; the old face pack is not loaded. Fresh sessions may still need software packages and the small tunnel binary.')
p('Keep the two pack outputs separate if your Kaggle output quota cannot hold both. Actual available disk is checked. Saving a Dataset does not automatically attach it to the GPU notebook. Startup never starts image generation. Keep the session URL private.','small')
link('Open studio notebook',BASE+'/blob/main/Image_Studio_Commercial_v7.ipynb')
link('Open CPU model-pack notebook',BASE+'/blob/main/Image_Studio_Commercial_Model_Packs.ipynb')

start('02','Build the shot and cast','Cinematic realism. A clear brief. The same people across changing worlds.')
step(1,'Write the shot brief','Choose an episode or <b>Custom</b>, then the intended format, cast and scene. Specify what the viewer must notice, the emotional beat and the lighting. World, palette and wing remain editable. Enter the precise scripture reference you intend; verify it before publishing.')
step(2,'Create the scene','Use <b>Create</b> with FLUX.2 Klein <b>4B distilled</b>. The graph starts at four steps and CFG 1. Use grounded materials, thoughtful film light and believable anatomy. Keep generated lettering out of the scene; add final text during delivery.')
step(3,'Approve the recurring leads','Use <b>Approve Cast</b> with a clear, suitable original reference. Inspect it, name the role, add a useful character prefix and explicitly set approved=true. Save the PNG and its cast record together, privately. Approval records a creative choice; it does not establish the image’s rights or upstream model history.')
h('Choose the smallest useful edit')
p('<b>Reference Edit:</b> broader prompt-directed changes.<br/><b>Masked Edit:</b> paint a white replacement region in LoadImage → MaskEditor. The model generates an edit, then the selected region is composited into the original, with a feather band.<br/><b>Approved Duet Identity Edit:</b> map target scene, Lead One and Lead Two references explicitly; review each face separately.')
box('The visual constitution','The human leads carry the anthology. Evil entities stay non-human and faceless. The Seeker stays faceless and cloaked. No anime, shame-based imagery or exploitative treatment of grief. Let each song have its own world.')
p('At CFG 1, the brief’s negative text is retained for review but is not active negative guidance. A generative edit may change clothing, hands or background as well as the requested detail.','small')

start('03','Choose the face method','Dedicated head replacement and prompt-directed identity editing solve different needs.')
box('First dedicated candidate: BFS Head Swap 4B','Use <b>BFS Head Swap 4B</b> or <b>Approved Cast BFS Head Swap 4B</b>. Picture 1 is the clean target scene. Picture 2 is the source identity. The dedicated 4B adapter starts at strength <b>1.0</b> on the reviewed Klein 4B base. It can replace the whole head, including hair.')
h('A focused instruction')
p('Use Picture 1 as the base scene. Replace its head with the head from Picture 2. Preserve the reference identity, eyes, nose and hair; match the target head rotation, gaze and expression. Keep the body, clothing, framing and background intact.')
h('When you want less than a head swap')
p('Try <b>Identity Edit</b> with a clear instruction naming the person to change. Use a masked-edit pass when the change must stay inside a smaller region. Generic identity editing is generative; it can drift from the exact reference. It is not an identity guarantee.')
h('Supplemental CPU option')
p('<b>Mesh Face Composite</b> transfers source pixels using Google landmarks. Use similar pose, expression and lighting. Face indices start at 0, left to right. Feather retains source features; optional Poisson blending can weaken identity contrast. The tested sample showed mouth and edge artifacts: this is a fallback, not the quality recommendation.')
h('Review before upscaling')
p('Compare eyes, nose, jaw, age, expression, hair boundary, skin tone and lighting with the approved photo. Inspect teeth, earrings, glasses and occlusion. For two people, verify the identity and placement of each lead independently.')
p('BFS’s author declares MIT for the adapter; the chosen 4B base is Apache 2.0. The 9B family has different terms and is excluded. No comparative likeness ranking or successful BFS/T4 inference has been established here.','small')

start('04','Upscale with intention','×4 multiplies each dimension. It does not simply set a “4K” switch.')
box('Your existing ×4 workflow','<b>1024 × 1024 → 4096 × 4096</b><br/>Real-ESRGAN x4plus performs the model upscale. If your starting image was around 1K per side, the result is around 4K per side. Keep that enlarged image as a reference master; avoid running ×4 over it again unless the larger dimensions are deliberate.')
h('Pick the finishing workflow')
p('<b>Upscale 2x:</b> the 4× model result is downsampled to 2× dimensions.<br/><b>Upscale 4x:</b> the full Real-ESRGAN enlargement.<br/><b>Restore SeedVR2:</b> a reconstruction alternative that can invent detail. Begin with a source around 1024 pixels or smaller, then inspect facial features and texture carefully.')
h('What to inspect at 100%')
p('Check eyes and teeth first, then hair, fabric, skin texture, lettering and high-contrast edges. Reject halos, waxy skin and invented details that change the person. Compare with the original at a similar display size; more pixels alone do not prove better likeness.')
h('Still resolution and clip resolution')
p('A 4K reference does not force a 4K clip. The video model may resize it for generation. Keep the first video render, finish the edit, then use your <b>movie upscaler</b> for the intended delivery resolution. Check moving edges, faces, flicker and audio sync afterward.')
box('Three separate files to keep','<b>Original</b> — the source image before enhancement.<br/><b>Reference master</b> — the approved enlarged still.<br/><b>Clip master</b> — the rendered video before movie upscaling.<br/>Keep final delivery files beside these, not in place of them.')
p('Real-ESRGAN’s BSD-3-Clause and SeedVR2’s Apache-2.0 licenses permit commercial use under their conditions. Those permissions do not replace the rights required for source images or likenesses.','small')

presets=json.loads((ROOT.parent/'image_studio_commercial/jjp_presets.json').read_text())['episodes']
eps=[(k,v) for k,v in presets.items() if k!='Custom']
for block in range(2):
 start(f'0{5+block}','The episode worlds' + (' · I' if block==0 else ' · II'),'The launch plan’s eight worlds, translated into practical image direction.')
 for title,ep in eps[block*4:block*4+4]:
  h(title)
  p('<b>'+escape(ep['wing'])+' · '+escape(ep['cast'])+'</b><br/>'+escape(ep['world']),gap=6)
  p('<b>Palette:</b> '+escape(ep['palette'])+'<br/><b>Reference:</b> '+escape(ep['reference'] or 'Choose and verify the intended passage; none is preset.'),'small',gap=17)
 p('Episode presets are starting briefs, not final publication approval. Add custom worlds and wings as the catalog grows. Verify references and any supplied scripture quotation against the intended passage.','small')

start('07','Deliver, archive, then animate','Keep a clean master and a production record for every accepted shot.')
step(1,'Finish the still before layout','Approve identity and scene detail, then upscale if needed. Load the finished still into <b>Delivery Export</b>. Creation workflows already include JJP export. Choose the desired layout and add the final title/reference as deterministic lettering.')
step(2,'Understand the two PNGs','<b>master.png</b> preserves the image entering export. It may already include a swap or upscale; it is not necessarily the clean original.<br/><b>delivery.png</b> contains the formatted result. Square, landscape and vertical layouts use padding/Lanczos resizing. That formatting is not neural upscaling.')
step(3,'Keep the episode record','Save <b>episode.json</b>, the PNG metadata, graph/prompt/seed, model provenance.json, license notices and relevant cast records. Save Kaggle outputs before ending the session. Model datasets alone do not preserve your private cast library or finished images.')
h('Suggested shot folder')
p('<b>01_originals</b> · identity photo and clean target<br/><b>02_working</b> · scene candidates and edits<br/><b>03_approved</b> · accepted reference stills<br/><b>04_delivery</b> · branded PNGs and episode record<br/><b>05_video</b> · initial clips and movie-upscaled finals<br/><b>06_provenance</b> · graphs, settings, model notices and rights records')
box('Before this shot enters the music video','Review likeness and dignity; check the antagonist/Seeker rules; verify the scripture reference and title spelling; inspect artifacts and framing. Once animated, check continuity and motion again. Export does not automatically approve publication.')

start('08','Know what is in the stack','A model-license record, with the practical limits stated plainly.')
h('The reviewed Commercial v7 profile')
for title,text in [('Apache 2.0','FLUX.2 Klein 4B distilled, Qwen3-4B encoder, Flux2 VAE, SeedVR2 weights and Google Face Landmarker components.'),('BSD 3-Clause','Real-ESRGAN x4plus; preserve its original notices when redistributing the model.'),('Author-declared MIT','The exact pinned BFS head v1 Klein 4B adapter. Its model card and family guide are retained with the pack.')]:p('<b>'+title+'</b> — '+text,gap=12)
h('The historical image profile')
p('InSwapper and buffalo_l pretrained models have research-only terms. Juggernaut’s extra commercial wording and FLUX 3’s custom terms require separate review; they are not blanket-labelled forbidden outputs. Commercial v7 replaces those defaults and uses a fresh installation folder. It does not retroactively license older processing.')
h('What was actually checked')
p('Twenty-four workflows loaded in a real ComfyUI browser session. CPU landmark compositing/export and tiny Real-ESRGAN inference ran. The exact BFS adapter hash was verified; the pinned ComfyUI parser recognised all 80 patches. Full FLUX/BFS/SeedVR2 GPU inference, T4 performance, likeness quality, live Kaggle/Cloudflare and actual Dataset upload were not verified in this environment.')
p('Sources and model declarations were reviewed on 9 November 2026. This companion was updated on 10 November to reflect your Flow face swaps and ×4 upscaling. No model is presented as a measured overall “best”.','small')
link('Read the full license audit',DOC+'LICENSE_AUDIT.md')
link('Read model comparisons and Pixaroma source review',DOC+'RESEARCH.md')
link('Read validation details',DOC+'VALIDATION.md')
link('Read the Kaggle model-pack instructions',DOC+'MODEL_PACKS.md')
link('Open all workflow files',BASE+'/tree/main/notebooks/image_studio_commercial/workflow_files')
c.save()
print(OUT,OUT.stat().st_size,'bytes;',page,'pages')
