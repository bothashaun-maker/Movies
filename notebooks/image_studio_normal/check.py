import ast,json,pathlib,unittest,os,urllib.request
R=pathlib.Path(__file__).parent
class Checks(unittest.TestCase):
 def test_notebook(self):
  n=json.loads(pathlib.Path(os.environ.get('STUDIO_NOTEBOOK',str(R/'Image_Studio_Normal_v10.ipynb'))).read_text());self.assertEqual(n['nbformat'],4)
  codes=[c for c in n['cells'] if c['cell_type']=='code'];self.assertEqual(len(codes),1)
  compile(''.join(codes[0]['source']),'notebook','exec')
  self.assertEqual(codes[0]['outputs'],[])
 def test_graph_edges(self):
  for name,w in json.loads((R/'workflows.json').read_text()).items():
   nodes={n['id']:n for n in w['nodes']}
   for lid,a,slot,b,inp,typ in w['links']:
    self.assertEqual(nodes[a]['outputs'][slot]['type'],typ)
    self.assertIn(nodes[b]['inputs'][inp]['type'],[typ,'*'])
    self.assertEqual(nodes[b]['inputs'][inp]['link'],lid)
    self.assertIn(lid,nodes[a]['outputs'][slot]['links'])
   for n in nodes.values():
    self.assertTrue(all(i['link'] or i.get('shape')==7 for i in n['inputs']))
   reached=set();todo=[n['id'] for n in nodes.values() if n['type'] in ['SaveImage','PreviewAny','JJPArtworkExport','JJPSaveApprovedCast']]
   while todo:
    i=todo.pop()
    if i in reached:continue
    reached.add(i);todo.extend(a for _,a,_,b,_,_ in w['links'] if b==i)
   self.assertEqual(reached,set(nodes),name)
 def test_native_node_definitions(self):
  info=json.load(urllib.request.urlopen(os.environ.get('COMFY_URL','http://127.0.0.1:8189')+'/object_info'))
  for w in json.loads((R/'workflows.json').read_text()).values():
   for n in w['nodes']:
    self.assertTrue(n['type'] in info,'Missing live node: '+n['type'])
    schema=info[n['type']]['input']
    for inp in n['inputs']:
     self.assertIn(inp['name'],{**schema.get('required',{}),**schema.get('optional',{})})
 def test_embedded_graphs(self):
  code=''.join(json.loads(pathlib.Path(os.environ.get('STUDIO_NOTEBOOK',str(R/'Image_Studio_Normal_v10.ipynb'))).read_text())['cells'][1]['source'])
  tree=ast.parse(code);v=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WORKFLOW_PAYLOAD' for t in n.targets))
  self.assertEqual(ast.literal_eval(v),json.loads((R/'workflows.json').read_text()))
if __name__=='__main__':unittest.main()
