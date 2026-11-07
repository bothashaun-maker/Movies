"""Full editable canvases, not a single opaque generation node."""
import copy

def next_canvas(base,title,model,lora,steps):
    g=copy.deepcopy(base);ns={n['id']:n for n in g['nodes']}
    ns[1]['type']='H3SuiteUNETLoader';ns[1]['widgets_values']=[model]
    ns[1]['title']=title+' · T4 ×2'
    ns[2].update(type='H3NextMolmo',widgets_values=[],title='Molmo2 · experimental T4 encoder')
    ns[3].update(type='H3SuiteVAELoader',widgets_values=['minimax_h3_video_vae_int8_convrot.safetensors','cuda:0'],title='Original H3 VAE · reference encoding')
    ns[4].update(type='H3SuiteVAELoader',title='Audio VAE · decode worker')
    ns[5].update(type='H3SuiteLoraLoader',widgets_values=[lora,1.0],title=f'{steps} step adapter')
    ns[6]['widgets_values']=[12.,3.]
    ns[7].update(type='H3NextReference',title='Prompt / size / seconds · optional reference',widgets_values=[ns[7]['widgets_values'][0],512,768,5.])
    ns[7]['inputs'][-1]['name']='reference_image'
    ns[7]['outputs'][0]['name']='positive'
    ns[10]['widgets_values']=['euler']
    ns[11]['widgets_values']=['simple',steps,1.]
    ns[16].update(type='H3SuiteSaveVideo',inputs=[{'name':'images','type':'IMAGE','link':None},{'name':'audio','type':'AUDIO','link':None}],widgets_values=['H3_Next/'+title.replace(' ','_')+'/clip',24.])
    ns[18]['title']='Upload reference image · connect IMAGE to reference_image'
    ns[18]['widgets_values']=['','image']
    sol=copy.deepcopy(ns[3]);sol.update(id=19,pos=[1790,760],widgets_values=['minimax_h3_sol_vae_int8_convrot.safetensors','cuda:0'],title='Sol-H3 INT8 · video decode worker')
    motion={'id':20,'type':'H3NextMotion','pos':[760,-170],'size':[310,110],'flags':{},'order':0,'mode':0,'inputs':[{'name':'model','type':'MODEL','link':None}],'outputs':[{'name':'MODEL','type':'MODEL','links':[]}],'properties':{},'widgets_values':[0.],'title':'Optional movement · 0 = off'}
    ns[19]=sol;ns[20]=motion;del ns[15]
    links=[l for l in g['links'] if l[1]!=15 and l[3]!=15]
    links=[l for l in links if not(l[1]==5 and l[3]==6) and not(l[1]==3 and l[3]==13)]
    links += [[0,5,0,20,0,'MODEL'],[0,20,0,6,0,'MODEL'],[0,19,0,13,1,'VAE'],[0,13,0,16,0,'IMAGE'],[0,14,0,16,1,'AUDIO']]
    for idx,n in enumerate(ns.values()):
        n['order']=idx;n['properties']={'Node name for S&R':n['type']}
        for i in n['inputs']:i['link']=None
        for o in n['outputs']:o['links']=[]
    for idx,l in enumerate(links,1):
        l[0]=idx;ns[l[1]]['outputs'][l[2]]['links'].append(idx);ns[l[3]]['inputs'][l[4]]['link']=idx
    g.update(nodes=list(ns.values()),links=links,last_node_id=20,last_link_id=len(links))
    g['extra'].update(h3_title=title,ds={'scale':.6,'offset':[100,240]})
    return g
