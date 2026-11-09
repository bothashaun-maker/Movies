import copy

def extend_workflows(originals,presets):
 def node(i,t,pos,widgets,ins,outs,title):
  return dict(id=i,type=t,pos=pos,size=[380,300],flags={},order=i,mode=0,inputs=[dict(name=a,type=b,link=None) for a,b in ins],outputs=[dict(name=a,type=b,links=[]) for a,b in outs],properties={'Node name for S&R':t},widgets_values=widgets,title=title)
 def connect(w,a,slot,b,name,typ,widget=False):
  nodes={n['id']:n for n in w['nodes']};n=nodes[b]
  found=next((j for j,i in enumerate(n['inputs']) if i['name']==name),None)
  if found is None:
   inp=dict(name=name,type=typ,link=None)
   if widget:inp['widget']={'name':name}
   n['inputs'].append(inp);found=len(n['inputs'])-1
  lid=max([e[0] for e in w['links']]+[0])+1
  n['inputs'][found]['link']=lid;nodes[a]['outputs'][slot].setdefault('links',[]).append(lid)
  w['links'].append([lid,a,slot,b,found,typ]);w['last_link_id']=lid
 additions={}
 for name,original in originals.items():
  w=copy.deepcopy(original);bid=max(n['id'] for n in w['nodes'])+1
  brief=node(bid,'JJPEpisodeBrief',[-530,0],['Custom','Episode key art','','Episode default','','','',True,'','',''],[],[('positive','STRING'),('negative','STRING'),('width','INT'),('height','INT'),('brief','STRING')],'JJP — episode, world, cast and delivery brief')
  brief['size']=[440,600];w['nodes'].append(brief);w['last_node_id']=bid
  for n in list(w['nodes']):
   if n['type']=='CLIPTextEncode':
    negative='Negative' in n.get('title','') or n['widgets_values']==['']
    if name!='Create FLUX3':connect(w,bid,1 if negative else 0,n['id'],'text','STRING',True)
   if n['type'] in ['EmptyLatentImage','EmptyFlux2LatentImage']:
    connect(w,bid,2,n['id'],'width','INT',True);connect(w,bid,3,n['id'],'height','INT',True)
   if name=='Create FLUX3' and n['id']==17:connect(w,bid,0,n['id'],'value','STRING',True)
   if n['type']=='SaveImage':
    n['type']='JJPArtworkExport';n['properties']={'Node name for S&R':'JJPArtworkExport'};n['title']='JJP — preserve master + branded delivery + episode record'
    n['inputs'][0]['name']='image';n['widgets_values']=['Original size' if name=='Face Swap' else 'From brief',name!='Face Swap',2048]
    n['outputs']=[dict(name='delivery',type='IMAGE',links=[]),dict(name='saved_files',type='STRING',links=[])]
    n['size']=[420,350];connect(w,bid,4,n['id'],'brief','STRING')
  w['extra']={'ds':{'scale':0.55,'offset':[750,250]}}
  additions['JJP - '+name]=w
 # Approved-library variant of the photo swap; manual upload variant is retained.
 cast=copy.deepcopy(additions['JJP - Face Swap']);source=next(n for n in cast['nodes'] if n['title'].startswith('SOURCE'))
 source.update(type='JJPLoadApprovedCast',properties={'Node name for S&R':'JJPLoadApprovedCast'},title='JJP — select approved identity reference',widgets_values=['Add approved cast reference first'],inputs=[],outputs=[dict(name='reference_image',type='IMAGE',links=source['outputs'][0]['links']),dict(name='character_prefix',type='STRING',links=[]),dict(name='cast_record',type='STRING',links=[])])
 brief=next(n for n in cast['nodes'] if n['type']=='JJPEpisodeBrief');export=next(n for n in cast['nodes'] if n['type']=='JJPArtworkExport')
 connect(cast,source['id'],1,brief['id'],'character_prefix','STRING',True);connect(cast,source['id'],2,export['id'],'cast_record','STRING')
 additions['JJP - Face Swap from Cast']=cast
 # Reference approvals are explicit operations, never inferred from a generated face.
 load=node(1,'LoadImage',[0,100],['','image'],[],[('IMAGE','IMAGE'),('MASK','MASK')],'Upload a reference you have approved')
 save=node(2,'JJPSaveApprovedCast',[450,100],['Lead One','Approved reference','',False],[('image','IMAGE')],[('reference','IMAGE'),('record','STRING')],'Inspect image, set role/prefix, then approve=true')
 preview=node(3,'PreviewAny',[900,100],[],[('source','*')],[('STRING','STRING')],'Approved cast record — save private outputs')
 approve=dict(nodes=[load,save,preview],links=[],last_node_id=3,last_link_id=0,groups=[],config={},extra={'ds':{'scale':0.7,'offset':[180,180]}},version=0.4)
 connect(approve,1,0,2,'image','IMAGE');connect(approve,2,1,3,'source','STRING')
 additions['JJP - Approve Cast']=approve
 # Two approved human identities, applied sequentially to chosen target indices.
 duet=copy.deepcopy(cast);nodes={n['id']:n for n in duet['nodes']}
 load1=nodes[source['id']];load1['title']='LEAD ONE — approved identity';b=next(n for n in duet['nodes'] if n['type']=='JJPEpisodeBrief');b['widgets_values'][3]='Duet'
 # Remove the single-reference prefix/record links; replace with the pair metadata.
 removed={e[0] for e in duet['links'] if e[1]==source['id'] and e[2] in [1,2]}
 duet['links']=[e for e in duet['links'] if e[0] not in removed]
 for n in duet['nodes']:
  for o in n['outputs']:o['links']=[i for i in o.get('links',[]) if i not in removed]
  for i in n['inputs']:
   if i.get('link') in removed:i['link']=None
 original_swap=next(n for n in duet['nodes'] if n['type']=='StudioFaceSwap');original_swap['title']='Swap Lead One — choose target index';export_node=next(n for n in duet['nodes'] if n['type']=='JJPArtworkExport')
 # Reroute export from the first swap to the final swap.
 old_lid=export_node['inputs'][0]['link'];duet['links']=[e for e in duet['links'] if e[0]!=old_lid]
 original_swap['outputs'][0]['links'].remove(old_lid);export_node['inputs'][0]['link']=None
 start=max(n['id'] for n in duet['nodes'])+1
 load2=node(start,'JJPLoadApprovedCast',[0,1000],['Add approved cast reference first'],[],[('reference_image','IMAGE'),('character_prefix','STRING'),('cast_record','STRING')],'LEAD TWO — approved identity')
 detect2=node(start+1,'StudioFaceDetect',[400,1000],[],[('image','IMAGE')],[('faces','STUDIO_FACES'),('detected_faces','STRING')],'Detect Lead Two')
 target2=node(start+2,'StudioFaceDetect',[1550,650],[],[('image','IMAGE')],[('faces','STUDIO_FACES'),('detected_faces','STRING')],'Re-detect after first swap')
 swap2=node(start+3,'StudioFaceSwap',[1950,450],[0,1],[('source_faces','STUDIO_FACES'),('target_faces','STUDIO_FACES')],[('IMAGE','IMAGE')],'Swap Lead Two — default target index 1')
 pair=node(start+4,'JJPCombineCast',[-500,750],[],[('lead_one','STRING'),('lead_two','STRING')],[('character_prefix','STRING'),('cast_record','STRING')],'Record both approved cast references')
 duet['nodes'].extend([load2,detect2,target2,swap2,pair]);duet['last_node_id']=start+4
 connect(duet,start,0,start+1,'image','IMAGE');connect(duet,original_swap['id'],0,start+2,'image','IMAGE');connect(duet,start+1,0,start+3,'source_faces','STUDIO_FACES');connect(duet,start+2,0,start+3,'target_faces','STUDIO_FACES');connect(duet,start+3,0,export_node['id'],'image','IMAGE')
 connect(duet,load1['id'],2,pair['id'],'lead_one','STRING');connect(duet,load2['id'],2,pair['id'],'lead_two','STRING');connect(duet,pair['id'],0,b['id'],'character_prefix','STRING',True);connect(duet,pair['id'],1,export_node['id'],'cast_record','STRING')
 export_node['pos']=[2350,450];additions['JJP - Duet Face Swap from Cast']=duet
 # Eight editable launch presets, using the same connected creation graph.
 for idx,title in enumerate(list(presets['episodes'])[1:],1):
  w=copy.deepcopy(additions['JJP - Create']);brief=next(n for n in w['nodes'] if n['type']=='JJPEpisodeBrief');brief['widgets_values'][0]=title
  additions[f'JJP Episode {idx:02d} - {title}']=w
 return additions
