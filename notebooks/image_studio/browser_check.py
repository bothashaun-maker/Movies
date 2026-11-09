import asyncio,json,pathlib,os
from playwright.async_api import async_playwright
R=pathlib.Path(__file__).parent
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
  page=await browser.new_page(viewport={'width':1800,'height':1100})
  errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
  await page.goto(os.environ.get('COMFY_URL','http://127.0.0.1:8189'),wait_until='networkidle')
  await page.wait_for_function('window.app && window.app.graph',timeout=30000)
  workflows=json.loads((R/'workflows.json').read_text())
  for name,wf in workflows.items():
   result=await page.evaluate('''async (wf)=>{
    await window.app.loadGraphData(wf);
    const nodes=window.app.graph._nodes.map(n=>({id:n.id,type:n.type,missing:!window.LiteGraph.registered_node_types[n.type],widgets:(n.widgets||[]).map(w=>({name:w.name,value:w.value}))}));
    const prompt=await window.app.graphToPrompt();return {nodes,prompt:prompt.output};
   }''',wf)
   api=result['prompt'];result=result['nodes']
   assert len(result)==len(wf['nodes']),(name,result)
   assert not any(n['missing'] for n in result),(name,result)
   for n in result:
    if n['type']=='UltimateSDUpscale':
     vals={w['name']:w.get('value') for w in n['widgets']}
     assert vals['denoise']==0.2 and vals['tile_width']==768 and vals['batch_size']==1,vals
    if n['type']=='KSampler' and name=='Create FLUX3':
     vals={w['name']:w.get('value') for w in n['widgets']};assert vals['steps']==16 and vals['cfg']==1,vals
   briefs=[n for n in result if n['type']=='JJPEpisodeBrief']
   if briefs:
    bid=str(briefs[0]['id'])
    for n in result:
     if n['type']=='JJPArtworkExport':assert api[str(n['id'])]['inputs']['brief']==[bid,4]
     if n['type'] in ['EmptyLatentImage','EmptyFlux2LatentImage']:
      assert api[str(n['id'])]['inputs']['width']==[bid,2]
      assert api[str(n['id'])]['inputs']['height']==[bid,3]
    if name.startswith('JJP Episode '):
     vals={w['name']:w.get('value') for w in briefs[0]['widgets']}
     assert vals['episode']==name.split(' - ',1)[1],vals
   print('Browser loaded:',name,len(result),'nodes')
   if name=='Upscale Creative':await page.screenshot(path=str(R.parent/'image-studio-v2-creative.png'))
  assert not errors,errors
  await browser.close()
asyncio.run(main())
