"""Adapt supplied canvases while retaining their native editing interfaces."""
import copy

def adapt_canvas(original,title,director=False):
    g=copy.deepcopy(original);nodes={n['id']:n for n in g['nodes']};links={l[0]:l for l in g['links']}
    sets={n['widgets_values'][0]:n for n in nodes.values() if n['type']=='SetNode'}
    bypass={'PixaromaFreeVram','PixaromaRunTimer','ModelAttentionBackend','MiniMaxLowVRAMAttention','MiniMaxChunkFeedForward'}
    def source(nid,slot):
        n=nodes[nid]
        if n['type']=='GetNode':
            n=sets[n['widgets_values'][0]];nid=n['id']
        if n['type']=='SetNode' or n['type'] in bypass:
            inp=next(i for i in n['inputs'] if i.get('link') is not None)
            l=links[inp['link']];return source(l[1],l[2])
        return nid,slot
    edges=[]
    remove=set()
    for n in nodes.values():
        if n['type'] in bypass|{'GetNode','SetNode','PixaromaMonitor'}:remove.add(n['id']);continue
        for i,inp in enumerate(n.get('inputs',[])):
            if inp.get('link') is not None:
                l=links[inp['link']];a,o=source(l[1],l[2]);edges.append([a,o,n['id'],i,l[5]])
    for n in nodes.values():
        kind=n['type'];w=n.get('widgets_values')
        if kind=='UNETLoader':
            n['type']='H3SuiteUNETLoader';n['widgets_values']=[w[0].replace('\\','/').split('/')[-1]]
        elif kind=='CLIPLoader':
            n['type']='H3SuiteCLIPLoader';n['widgets_values']=[w[0].replace('\\','/').split('/')[-1],'minimax','cuda:1']
        elif kind=='VAELoader':
            n['type']='H3SuiteVAELoader';n['widgets_values']=[w[0].replace('\\','/').split('/')[-1],'cuda:1' if 'audio' in w[0] else 'cuda:0']
        elif kind=='LoraLoaderModelOnly':n['type']='H3SuiteLoraLoader'
        elif kind=='MiniMaxH3PDDAccApply':n['type']='H3SuitePDDApply'
        elif kind=='KSamplerSelect' and 'PDD' in title:n['widgets_values']=['euler']
        elif kind=='PixaromaLoraLoader':
            for item in w[0]['loras']:item['name']=item['name'].replace('\\','/').split('/')[-1]
        elif kind=='PixaromaDuration':n['properties']['durationState']['seconds']=5
        elif kind=='PixaromaLongestSide':n['properties']['longestSideState'].update(size=768,ratio='2:3')
        elif kind=='PixaromaSizes':w[0].update(w=512,h=768)
        elif kind=='PixaromaDropdown':
            state=n['properties']['dropdownState'];state['index']=next((j for j,x in enumerate(state['options']) if x.get('v',['',''])[1]=='4'),0)
        elif kind=='MiniMaxH3DirectorCS':
            # Keep real Director UI. Start with a clean, editable five-second timeline.
            import json
            timeline=json.loads(w[6]);timeline.update(normalStartFrame=0,normalDurationFrames=120,reference_mode='OFF',global_prompt='A calm cinematic landscape at sunrise.',summary='A five-second landscape shot.',subjects=[],segments=[{'id':'h3-t4-start','start':0,'length':120,'prompt':'A slow camera move across the landscape.'}])
            # Remove media specific to the author's local workstation.
            for key in ('audioClips','audio_clips','motionClips','motion_clips','videoClips','images','imageTrack','audioTrack','motionTrack'):
                if key in timeline:timeline[key]=[]
            w[0:7]=[0,5,5,0,120,120,json.dumps(timeline)]
            # Widget order is from pinned Director schema: custom_width/custom_height.
            w[15:17]=[512,768]
            w[7]='A slow camera move across the landscape.';w[8]='120'
            timeline['overall_soundscape']='Gentle natural ambience.';timeline['non_diegetic_music']='';timeline['retakeMode']=False
            w[6]=json.dumps(timeline)
            n['properties'].update(global_prompt=timeline['global_prompt'],timeline_data=w[6],local_prompts=w[7],segment_lengths='120',
                start_second=0,end_second=5,duration_seconds=5,start_frame=0,end_frame=120,duration_frames=120,
                custom_width=512,custom_height=768,retakeMode=False)
        if n['type']!=kind:
            n['properties']={'Node name for S&R':n['type']}
    # Preserve source frames/audio wiring, replacing only the terminal save operation.
    save=next(n for n in nodes.values() if n['type'] in ('PixaromaSaveMp4','VHS_VideoCombine'))
    input_names={i['name']:j for j,i in enumerate(save['inputs'])}
    frame_slot=input_names.get('video_frames',input_names.get('images'))
    audio_slot=input_names.get('audio')
    incoming={e[3]:e for e in edges if e[2]==save['id']}
    newedges=[]
    for slot,newslot in ((frame_slot,0),(audio_slot,1)):
        if slot in incoming:
            e=incoming[slot];newedges.append([e[0],e[1],save['id'],newslot,'IMAGE' if newslot==0 else 'AUDIO'])
    edges=[e for e in edges if e[2]!=save['id']]+newedges
    save.update(type='H3SuiteSaveVideo',inputs=[{'name':'images','type':'IMAGE','link':None},{'name':'audio','type':'AUDIO','link':None}],outputs=[],widgets_values=[title.replace(' ','_')+'/clip',24.0],properties={'Node name for S&R':'H3SuiteSaveVideo'},title='Save MP4 · isolated T4 workers')
    g['nodes']=[n for n in nodes.values() if n['id'] not in remove]
    g['links']=[]
    for n in g['nodes']:
        for inp in n.get('inputs',[]):inp['link']=None
        for out in n.get('outputs',[]):out['links']=[]
    for a,o,b,i,t in edges:
        if a in remove or b in remove:continue
        ident=len(g['links'])+1;g['links'].append([ident,a,o,b,i,t])
        nodes[a]['outputs'][o]['links'].append(ident);nodes[b]['inputs'][i]['link']=ident
    g['last_link_id']=len(g['links']);g.setdefault('extra',{})['h3_title']=title
    return g
