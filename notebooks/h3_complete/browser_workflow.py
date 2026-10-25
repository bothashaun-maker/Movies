def browser_workflow(prompt,seed=42,megapixels=0.4,duration=5.0):
    return {
        'last_node_id':1,'last_link_id':0,
        'nodes':[{'id':1,'type':'H3KaggleRender','pos':[180,120],'size':[520,440],
                  'flags':{},'order':0,'mode':0,
                  'inputs':[{'name':'image','type':'IMAGE','link':None}],
                  'outputs':[], 'properties':{'Node name for S&R':'H3KaggleRender'},
                  'widgets_values':[prompt,seed,'fixed',megapixels,duration,'portrait'],
                  'title':'MiniMax H3 — 4 steps · T4 ×2 · native audio'}],
        'links':[],'groups':[],'config':{},'extra':{},'version':0.4,
    }
