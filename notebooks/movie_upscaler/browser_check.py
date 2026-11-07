"""Load both real canvases and verify the actual exported API inputs."""
import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).parent
URL=os.environ.get('COMFY_URL','http://127.0.0.1:8188')

async def main():
    graphs=json.loads((ROOT/'workflows.json').read_text())
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('CHROMIUM','/usr/bin/chromium'),headless=True,args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':1600,'height':1000})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.goto(URL+'/?movie_studio=v1')
        await page.wait_for_function("() => !!window.LiteGraph?.registered_node_types?.MovieStudioChunk && !!window.LiteGraph?.registered_node_types?.SeedVR2VideoUpscaler",timeout=90000)
        await page.wait_for_function("() => { try { return !!window.comfyAPI.app.app.canvas?.canvas; } catch { return false; } }",timeout=90000)
        await page.wait_for_timeout(500)
        await page.wait_for_function("() => window.comfyAPI.app.app.graph._nodes.some(n => n.type === 'SeedVR2VideoUpscaler') && !new URL(location.href).searchParams.has('movie_studio')",timeout=90000)
        startup_errors=list(errors); errors.clear()
        exported={}
        for name,graph in graphs.items():
            result=await page.evaluate('''async graph => {
                const {app}=await import('/scripts/app.js');
                await app.loadGraphData(graph,true,true,'Movie Studio test.json');
                await new Promise(r=>setTimeout(r,500));
                const serialized=await app.graphToPrompt();
                return {api:serialized.output,nodes:app.graph._nodes.map(n=>({type:n.type,widgets:n.widgets?.map(w=>({name:w.name,value:w.value}))}))};
            }''',graph)
            assert len(result['nodes'])==len(graph['nodes']),result
            assert all(n['type'] in ['MovieStudioChunk','MovieStudioCommit','SeedVR2LoadDiTModel','SeedVR2LoadVAEModel','SeedVR2VideoUpscaler','UpscaleModelLoader','ImageUpscaleWithModel','ImageScaleBy'] for n in result['nodes'])
            if 'SeedVR2' in name:
                up=next(n['inputs'] for n in result['api'].values() if n['class_type']=='SeedVR2VideoUpscaler')
                assert up['batch_size']==5 and up['uniform_batch_size'] is True,up
                assert up['color_correction']=='lab' and up['temporal_overlap']==2,up
                assert isinstance(up['resolution'],list),up
            else:
                resize=next(n['inputs'] for n in result['api'].values() if n['class_type']=='ImageScaleBy')
                assert resize['scale_by']==0.5,resize
            exported[name]=result['api']
            print(name,json.dumps(result,indent=1))
        assert not errors,errors
        if startup_errors: print('Upstream frontend startup diagnostics:',startup_errors)
        out=Path(os.environ.get('MOVIE_CHECK_OUTPUT','/tmp/movie-api-graphs.json'))
        out.write_text(json.dumps(exported,indent=2))
        await browser.close()

if __name__=='__main__':asyncio.run(main())
