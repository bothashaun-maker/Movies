import unittest,tempfile,pathlib,sys,importlib.util,json,hashlib
import numpy as np
import torch
from PIL import Image
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R.parent/'ComfyUI'))
spec=importlib.util.spec_from_file_location('jjp',R/'jjp_nodes.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);self.old=m.folder_paths.get_output_directory;m.folder_paths.get_output_directory=lambda:str(self.root)
  self.cast_roots=m.cast_roots;m.cast_roots=lambda:[self.root/'JJP/cast']
  self.image=torch.linspace(0,1,300*200*3).reshape((1,200,300,3))
 def tearDown(self):m.folder_paths.get_output_directory=self.old;m.cast_roots=self.cast_roots;self.tmp.cleanup()
 def brief(self,**kwargs):
  args=dict(episode='Verdict of the Lamb',format='Square cover',scene='Two figures in a doorway.',cast='Episode default',character_prefix='',title='',scripture_reference='',use_episode_reference=True);args.update(kwargs);return m.JJPEpisodeBrief().brief(**args)
 def test_eight_episode_worlds_and_default_cast(self):
  self.assertEqual(len(m.PRESETS['episodes'])-1,8)
  for name,p in m.PRESETS['episodes'].items():
   out=self.brief(episode=name);d=json.loads(out[4]);self.assertEqual(d['cast'],p['cast']);self.assertIn('cinematic',out[0]);self.assertIn('faceless',out[0]);self.assertEqual((out[2],out[3]),(1024,1024))
 def test_future_custom_wing(self):
  out=self.brief(episode='Custom',custom_world='A midnight jazz garden',custom_palette='cobalt and amber',custom_wing='Jazz / Night')
  d=json.loads(out[4]);self.assertEqual(d['wing'],'Jazz / Night');self.assertIn('cobalt and amber',out[0]);self.assertIn('midnight jazz garden',out[0])
 def test_explicit_reference_never_rewritten(self):
  out=self.brief(scripture_reference='User-entered reference');self.assertEqual(json.loads(out[4])['scripture_reference'],'User-entered reference')
  self.assertEqual(json.loads(self.brief(use_episode_reference=False)[4])['scripture_reference'],'')
 def test_cast_requires_explicit_approval(self):
  with self.assertRaises(ValueError):m.JJPSaveApprovedCast().save(self.image,'Lead One','ref','prefix',False)
  self.assertEqual(m.cast_records(),{})
 def test_cast_roundtrip_and_tamper_check(self):
  m.JJPSaveApprovedCast().save(self.image,'Lead One','ref','approved clothing prefix',True)
  records=m.cast_records();self.assertEqual(len(records),1);key=next(iter(records));out=m.JJPLoadApprovedCast().load(key)
  self.assertEqual(out[1],'approved clothing prefix');self.assertEqual(out[0].shape,self.image.shape)
  path=records[key][1];Image.new('RGB',(300,200),'red').save(path)
  with self.assertRaisesRegex(ValueError,'changed'):m.JJPLoadApprovedCast().load(key)
 def test_export_keeps_master_and_exact_reference(self):
  out=m.JJPArtworkExport().export(self.image,self.brief()[4],'Square cover',True,512,prompt={'test':'graph'},extra_pnginfo={'workflow':{'nodes':[]}})
  roots=list((self.root/'JJP/episodes').glob('*/*'));self.assertEqual(len(roots),1);root=roots[0]
  master=Image.open(root/'master.png');delivery=Image.open(root/'delivery.png');self.addCleanup(master.close);self.addCleanup(delivery.close);d=json.loads((root/'episode.json').read_text())
  self.assertEqual(master.size,(300,200));self.assertEqual(delivery.size,(512,512));self.assertEqual(d['scripture_reference'],'John 19:30')
  self.assertTrue(np.array_equal(np.asarray(master),(self.image[0].numpy()*255).round().astype('uint8')))
  self.assertEqual(json.loads(master.info['prompt']),{'test':'graph'});self.assertEqual(d['approval_status'].split(';')[0],'Needs human review')
  self.assertEqual(out['result'][0].shape,(1,512,512,3))
 def test_raw_export_and_unique_output_dirs(self):
  for _ in range(2):m.JJPArtworkExport().export(self.image,self.brief()[4],'Original size',False,512)
  roots=list((self.root/'JJP/episodes').glob('*/*'));self.assertEqual(len(roots),2)
  for root in roots:self.assertEqual((root/'master.png').read_bytes(),(root/'delivery.png').read_bytes())
 def test_duet_metadata_requires_correct_approved_roles(self):
  one={'role':'Lead One','prompt_prefix':'one'};two={'role':'Lead Two','prompt_prefix':'two'}
  prefix,data=m.JJPCombineCast().combine(json.dumps(one),json.dumps(two));self.assertIn('one',prefix);self.assertEqual(len(json.loads(data)['references']),2)
  with self.assertRaises(ValueError):m.JJPCombineCast().combine(json.dumps(two),json.dumps(one))
 def test_all_delivery_ratios(self):
  for name,ratio in [('Square cover',1),('YouTube 16:9',16/9),('Vertical 9:16',9/16)]:
   out=m.JJPArtworkExport().export(self.image,self.brief()[4],name,False,1024)['result'][0]
   self.assertAlmostEqual(out.shape[2]/out.shape[1],ratio,delta=.002)
if __name__=='__main__':unittest.main()
