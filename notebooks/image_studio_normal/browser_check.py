import asyncio,json,os
from pathlib import Path
from playwright.async_api import async_playwright
R=Path(__file__).parent
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox']);page=await browser.new_page(viewport={'width':1800,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(os.environ.get('COMFY_URL','http://127.0.0.1:8189'),wait_until='networkidle');await page.wait_for_function('window.app && window.app.graph',timeout=30000)
  for name,w in json.loads((R/'workflows.json').read_text()).items():
   result=await page.evaluate('''async wf=>{await window.app.loadGraphData(wf);return {nodes:window.app.graph._nodes.map(n=>({id:n.id,type:n.type,missing:!window.LiteGraph.registered_node_types[n.type],widgets:(n.widgets||[]).map(w=>({name:w.name,value:w.value}))})),prompt:(await window.app.graphToPrompt()).output}}''',w)
   assert len(result['nodes'])==len(w['nodes']);assert not any(n['missing'] for n in result['nodes']);api=result['prompt']
   assert not any(n['type'].startswith('JJP') for n in result['nodes'])
   for n in result['nodes']:
    inputs=api[str(n['id'])]['inputs']
    if n['type']=='CLIPTextEncode':assert isinstance(inputs['text'],str) and inputs['text'],(name,inputs)
    if n['type']=='Flux2Scheduler':assert inputs['steps']==4
    if n['type']=='CFGGuider':assert inputs['cfg']==1.0
    if n['type']=='LoraLoaderModelOnly':assert inputs['lora_name']=='bfs_head_v1_flux-klein_4b.safetensors' and inputs['strength_model']==1.0
    if n['type']=='PrimitiveInt':assert inputs['value'] in (1216,832),(name,inputs)
    if n['type']=='EmptyFlux2LatentImage':assert isinstance(inputs['width'],list) and isinstance(inputs['height'],list)
   print('Browser passed:',name,len(result['nodes']),'native/normal nodes')
   if name=='Create - Cinematic':
    await page.wait_for_timeout(1800)
    await page.screenshot(path=str(R/'create-preview.png'))
  assert not errors,errors;await browser.close()
asyncio.run(main())
