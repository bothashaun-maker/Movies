import unittest,json,tempfile,ast
from pathlib import Path
from workflow_install import install_workflows
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def setUp(self):self.workflows=json.loads((R/'workflows.json').read_text())
 def test_only_normal_workflows_and_png_save(self):
  self.assertEqual(len(self.workflows),10)
  for name,w in self.workflows.items():
   self.assertFalse(any(t in json.dumps(w).lower() for t in ['jjp','episode','scripture','approved cast','lead one','lead two']))
   self.assertEqual(sum(n['type']=='SaveImage' for n in w['nodes']),1,name)
 def test_model_and_face_support_unchanged(self):
  old=R.parent/'image_studio_commercial'
  for f in ['catalog.json','face_nodes.py','startup_models.py'] :self.assertEqual((R/f).read_bytes(),(old/f).read_bytes(),f)
 def test_original_upscale_graphs_are_unchanged(self):
  old=json.loads((R.parent/'image_studio/workflows.json').read_text())
  for name in ['Upscale 2x','Upscale 4x','Restore SeedVR2']:self.assertEqual(self.workflows[name],old[name])
 def test_prompt_and_dimensions_are_plain_native_controls(self):
  w=self.workflows['Create - Cinematic'];self.assertEqual(sum(n['type']=='PrimitiveInt' for n in w['nodes']),2)
  p=next(n for n in w['nodes'] if n['type']=='CLIPTextEncode');self.assertTrue(p['widgets_values'][0]);self.assertEqual([i['name'] for i in p['inputs']],['clip'])
 def test_runtime_does_not_enable_production_nodes_or_builder(self):
  s=(R/'launcher.py').read_text();self.assertNotIn('JJP_NODE_SOURCE',s);self.assertNotIn('custom_jjp=',s);self.assertNotIn('PACK_BUILDER_SOURCE',s)
  tree=ast.parse(s);spawn=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='spawn' and isinstance(n.args[0],ast.List) and any(isinstance(x,ast.Constant) and x.value=='--whitelist-custom-nodes' for x in n.args[0].elts))
  self.assertNotIn('jjp_image_production',[x.value for x in spawn.args[0].elts if isinstance(x,ast.Constant)])
 def test_old_menus_archived_without_touching_user_graphs(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'workflows';folder.mkdir();(folder/'Image Studio - JJP Episode - old.json').write_text('old graph');(folder/'My own graph.json').write_text('personal')
   install_workflows(folder,self.workflows,root/'archive');self.assertFalse((folder/'Image Studio - JJP Episode - old.json').exists());self.assertEqual((folder/'My own graph.json').read_text(),'personal');self.assertEqual(next((root/'archive').iterdir()).read_text(),'old graph')
   self.assertEqual(len(list(folder.glob('Image Studio - *.json'))),10)
if __name__=='__main__':unittest.main()
