"""Load both real canvases and verify the actual exported API inputs."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
from playwright.async_api import async_playwright

ROOT=Path(__file__).parent
URL=os.environ.get('COMFY_URL','http://127.0.0.1:8188')

async def main():
    graphs=json.loads((ROOT/'workflows.json').read_text())
    fixture=Path(os.environ.get('MOVIE_UPLOAD_FIXTURE','/workspace/work/movie-upload-fixture.mp4'))
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=24x16:rate=24',
                    '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-frames:v','3','-t','0.125',
                    '-c:v','libx264','-c:a','aac','-metadata',f'comment=browser-upload-{time.time_ns()}',str(fixture)],check=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('CHROMIUM','/usr/bin/chromium'),headless=True,args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':1600,'height':1000})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.goto(URL+'/?movie_studio=v2')
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
                return app.graph._nodes.map(n=>({type:n.type,widgets:n.widgets?.map(w=>({name:w.name,value:w.value}))}));
            }''',graph)
            button=await page.evaluate('''() => {
                const app=window.comfyAPI.app.app;
                const node=app.graph._nodes.find(n=>n.type==='MovieStudioUpload');
                const widget=node.widgets.find(w=>w.name==='Upload clip');
                const rect=app.canvas.canvas.getBoundingClientRect(), ds=app.canvas.ds;
                return {x:rect.x+(node.pos[0]+node.size[0]/2+ds.offset[0])*ds.scale,
                        y:rect.y+(node.pos[1]+widget.last_y+10+ds.offset[1])*ds.scale};
            }''')
            assert 55 < button['x'] < 1500 and 100 < button['y'] < 900,button
            async with page.expect_file_chooser() as chooser:
                await page.mouse.click(button['x'],button['y'])
            await (await chooser.value).set_files(str(fixture))
            await page.wait_for_function("() => window.comfyAPI.app.app.graph._nodes.find(n=>n.type==='MovieStudioUpload')?.widgets.find(w=>w.name==='source_path')?.value?.includes('movie-clips/')",timeout=60000)
            result=await page.evaluate('''async () => {
                const {app}=await import('/scripts/app.js');
                const serialized=await app.graphToPrompt();
                return {api:serialized.output,nodes:app.graph._nodes.map(n=>({type:n.type,widgets:n.widgets?.map(w=>({name:w.name,value:w.value}))}))};
            }''')
            assert len(result['nodes'])==len(graph['nodes']),result
            assert all(n['type'] in ['MovieStudioUpload','MovieStudioChunk','MovieStudioCommit','SeedVR2LoadDiTModel','SeedVR2LoadVAEModel','SeedVR2VideoUpscaler','UpscaleModelLoader','ImageUpscaleWithModel','ImageScaleBy'] for n in result['nodes'])
            uploaded=next(n['inputs']['source_path'] for n in result['api'].values() if n['class_type']=='MovieStudioUpload')
            assert Path(uploaded).read_bytes()==fixture.read_bytes()
            commit=next(n['inputs'] for n in result['api'].values() if n['class_type']=='MovieStudioCommit')
            assert commit['export_format']=='lossless_mkv' and commit['crf']==14 and commit['preset']=='slow',commit
            if 'SeedVR2' in name:
                up=next(n['inputs'] for n in result['api'].values() if n['class_type']=='SeedVR2VideoUpscaler')
                assert up['batch_size']==5 and up['uniform_batch_size'] is True,up
                assert up['color_correction']=='lab' and up['temporal_overlap']==2,up
                assert isinstance(up['resolution'],list),up
                vae=next(n['inputs'] for n in result['api'].values() if n['class_type']=='SeedVR2LoadVAEModel')
                assert vae['encode_tile_overlap']==vae['decode_tile_overlap']==128,vae
                assert vae['device']=='cuda:1',vae
            else:
                resize=next(n['inputs'] for n in result['api'].values() if n['class_type']=='ImageScaleBy')
                assert isinstance(resize['scale_by'],list),resize
                source=next(k for k,n in result['api'].items() if n['class_type']=='MovieStudioChunk')
                assert resize['scale_by']==[source,3],resize
            exported[name]=result['api']
            if os.environ.get('MOVIE_SCREENSHOT_DIR'):
                await page.screenshot(path=str(Path(os.environ['MOVIE_SCREENSHOT_DIR'])/('movie-seed-v2.png' if 'SeedVR2' in name else 'movie-esrgan-v2.png')))
            print(name,json.dumps(result,indent=1))
        assert not errors,errors
        if startup_errors: print('Upstream frontend startup diagnostics:',startup_errors)
        out=Path(os.environ.get('MOVIE_CHECK_OUTPUT','/tmp/movie-api-graphs.json'))
        out.write_text(json.dumps(exported,indent=2))
        await browser.close()

if __name__=='__main__':asyncio.run(main())
