"""CPU-only size checks using pinned control-node helpers, before model downloads."""
def suite_helper(relative):
    import importlib.util
    path=COMFY/'custom_nodes/ComfyUI-Pixaroma/nodes'/relative
    spec=importlib.util.spec_from_file_location('suite_'+path.stem,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def suite_preflight(graph):
    from PIL import Image,ImageOps
    def image_size(link):
        node=graph[link[0]]
        if node['class_type'] not in ('LoadImage','PixaromaLoadImageMini'):raise ValueError('Use an image loader for the size reference.')
        name=node['inputs']['image'];p=(COMFY/'input'/name).resolve()
        if not p.is_relative_to((COMFY/'input').resolve()):raise ValueError('Image must be inside ComfyUI input.')
        with Image.open(p) as im:return ImageOps.exif_transpose(im).size
    def value(v):
        if not isinstance(v,list):return v
        node=graph[v[0]];i=node['inputs'];kind=node['class_type'];slot=v[1]
        if kind=='PixaromaDuration':return suite_helper('_duration_helpers.py').compute(i['DurationState'])[slot]
        if kind=='PixaromaSizes':
            state=json.loads(i['SizesState']);return int(state['w' if slot==0 else 'h'])
        if kind=='PixaromaLongestSide':
            size=suite_helper('_longest_side_helpers.py').compute(*image_size(i['image']),json.loads(i['LongestSideState']))['size'];return size[slot-1]
        raise ValueError('Unsupported size/duration source: '+kind)
    def check(w,h,f):
        if not (32<=w<=1024 and 32<=h<=1024 and w%32==h%32==0 and w*h<=430000 and 5<=f<=124 and f%17==5):raise ValueError('T4 limits: dimensions divisible by 32, at most 0.43 MP, duration at most 124 frames. Lower the size/duration controls.')
    for node in graph.values():
        i=node['inputs'];kind=node['class_type']
        if kind=='PixaromaLoadAudio':
            state=json.loads(i.get('LoadAudioState','{}'))
            if not state.get('file'):raise ValueError('Choose your audio recording in the Load Audio node before pressing Run.')
        if kind in ('MiniMaxH3ImageToVideo','MiniMaxH3ReferenceToVideo','MiniMaxH3DistilledImageToVideo'):
            check(int(value(i['width'])),int(value(i['height'])),int(value(i['length'])))
        elif kind=='MiniMaxH3DirectorCS':
            timeline=json.loads(i['timeline_data']);frames=int(i['duration_frames'])
            if timeline.get('retakeMode'):frames=int(timeline.get('retakeLength',frames))
            fps=float(i.get('frame_rate',24));seconds=float(value(i['duration'])) if 'duration' in i else frames/fps
            if 'end' in i:seconds=float(value(i['end']))-float(value(i.get('start',0)))
            f=round(seconds*24);f+=(5-f%17)%17
            check(int(value(i.get('width',i.get('custom_width',0)))),int(value(i.get('height',i.get('custom_height',0)))),f)
