import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from fractions import Fraction
from unittest import mock

SPEC = importlib.util.spec_from_file_location('media', Path(__file__).with_name('media.py'))
media = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media)

def make_video(path, rate='24', frames=13, extra=None):
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'testsrc2=size=24x16:rate={rate}',
           '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-map', '0:v', '-map', '1:a',
           '-frames:v', str(frames), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-t', str(frames/float(Fraction(rate)))]
    subprocess.run(cmd+(extra or [])+[str(path)], check=True)

class MediaChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source.mp4'
        make_video(self.source)
    def tearDown(self):
        self.temp.cleanup()
    def job(self, **kw):
        args = dict(source=self.source, root=self.root/'jobs', settings={'scale': 2}, core_frames=5, context=2, scale=2, max_edge=1920)
        args.update(kw)
        return media.prepare(**args)
    def complete(self, meta):
        # Deterministic nearest-neighbour fixture verifies chunk IO, not a model.
        import numpy as np
        while (index := media.next_chunk(meta)) is not None:
            raw, n = media.read_chunk(meta, index)
            frames = np.frombuffer(raw, dtype='uint8').reshape(n,16,24,3)
            up = frames.repeat(2, axis=1).repeat(2, axis=2)
            media.save_chunk(meta, index, up.tobytes(), 48, 32, n)
    def test_resume_exact_frames_and_audio(self):
        meta = self.job()
        self.assertEqual(media.next_chunk(meta), 0)
        self.complete(meta)
        # Reopen same source/settings: no restored frames need repeating.
        resumed = self.job()
        self.assertIsNone(media.next_chunk(resumed))
        for container in ['mp4', 'mkv']:
            target = media.export(resumed, container)
            data = media.probe(target, count=True)
            video = next(x for x in data['streams'] if x['codec_type']=='video')
            audio = next(x for x in data['streams'] if x['codec_type']=='audio')
            self.assertEqual(int(video['nb_read_frames']),13)
            self.assertEqual((video['width'],video['height']),(48,32))
            self.assertEqual(Fraction(video['avg_frame_rate']),24)
            self.assertAlmostEqual(float(video.get('start_time',0)), float(audio.get('start_time',0)), delta=0.05)
            # Same compressed audio packets survive stream-copy byte-for-byte.
            def audio_hash(path):
                return media.run(['ffmpeg','-v','error','-i',path,'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','-'])
            self.assertEqual(audio_hash(self.source),audio_hash(target))
            self.assertEqual(media.export(resumed,container), target)
    def test_context_frame_boundaries(self):
        meta = self.job()
        self.assertEqual(media.bounds(meta,0),(0,5,0,7))
        self.assertEqual(media.bounds(meta,1),(5,10,3,12))
        self.assertEqual(media.bounds(meta,2),(10,13,8,13))
        self.complete(meta)
        # Decode each checkpoint, then compare against full source enlarged.
        import numpy as np
        source = media.run(['ffmpeg','-v','error','-i',self.source,'-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
        expected = np.frombuffer(source,dtype='uint8').reshape(13,16,24,3).repeat(2,1).repeat(2,2).tobytes()
        actual = b''.join(media.run(['ffmpeg','-v','error','-i',Path(meta['job'])/f'chunk-{i:06d}.mkv','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-']) for i in range(3))
        self.assertEqual(actual,expected)
    def test_corrupt_checkpoint_is_preserved(self):
        meta=self.job();self.complete(meta)
        target=Path(meta['job'])/'chunk-000000.mkv'
        target.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError,'Corrupt committed'):
            media.next_chunk(meta)
        self.assertEqual(target.read_bytes(),b'corrupt')
    def test_incomplete_and_wrong_geometry_not_exported(self):
        meta=self.job()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            media.export(meta)
        with self.assertRaisesRegex(ValueError,'wrong target'):
            media.save_chunk(meta,0,b'',24,16,7)
        self.assertFalse((Path(meta['job'])/'chunk-000000.json').exists())
    def test_target_and_disk_preflight(self):
        with self.assertRaisesRegex(ValueError,'exceeds'):
            self.job(max_edge=32)
        usage=type('Usage',(),{'free':1})()
        with mock.patch.object(media.shutil,'disk_usage',return_value=usage):
            with self.assertRaisesRegex(RuntimeError,'Not enough disk'):
                self.job()
    def test_fractional_rate_and_silent_video(self):
        source=self.root/'silent.mp4'
        media.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=24x16:rate=30000/1001','-frames:v','13','-an','-c:v','libx264',source])
        meta=self.job(source=source)
        self.assertEqual(meta['info']['fps'],'30000/1001')
        self.complete(meta)
        result=media.probe(media.export(meta),count=True)
        self.assertEqual(len(result['streams']),1)
        self.assertEqual(Fraction(result['streams'][0]['avg_frame_rate']),Fraction(30000,1001))
    def test_changed_settings_separate_job(self):
        self.assertNotEqual(self.job()['job'],self.job(settings={'scale':2,'seed':43})['job'])
    def test_seek_with_nonzero_start_and_audio_offset(self):
        shifted=self.root/'shifted.mkv'
        make_video(shifted,frames=37,extra=['-g','12','-output_ts_offset','5'])
        meta=self.job(source=shifted)
        self.assertAlmostEqual(meta['info']['start'],5,delta=0.03)
        self.complete(meta)
        target=media.export(meta,'mkv')
        def offset(path):
            streams=media.probe(path)['streams']
            video=next(s for s in streams if s['codec_type']=='video')
            audio=next(s for s in streams if s['codec_type']=='audio')
            return float(audio['start_time'])-float(video['start_time'])
        self.assertAlmostEqual(offset(target),offset(shifted),delta=0.03)
        import numpy as np
        full=media.run(['ffmpeg','-v','error','-i',shifted,'-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
        expected=np.frombuffer(full,dtype='uint8').reshape(37,16,24,3).repeat(2,1).repeat(2,2).tobytes()
        actual=b''.join(media.run(['ffmpeg','-v','error','-i',Path(meta['job'])/f'chunk-{i:06d}.mkv','-f','rawvideo','-pix_fmt','rgb24','-threads','1','-']) for i in range(meta['chunks']))
        self.assertEqual(actual,expected)
    def test_mkv_multiple_audio_and_subtitles(self):
        sub=self.root/'caption.srt'
        sub.write_text('1\n00:00:00,000 --> 00:00:00,400\nTest caption\n')
        source=self.root/'tracks.mkv'
        media.run(['ffmpeg','-v','error','-y','-i',self.source,'-i',sub,'-map','0:v','-map','0:a','-map','0:a','-map','1:s','-c','copy',source])
        meta=self.job(source=source)
        self.complete(meta)
        streams=media.probe(media.export(meta,'mkv'))['streams']
        self.assertEqual(sum(s['codec_type']=='audio' for s in streams),2)
        self.assertEqual(sum(s['codec_type']=='subtitle' for s in streams),1)
    def test_vfr_rejected(self):
        target=self.root/'variable.mkv'
        media.run(['ffmpeg','-v','error','-y','-i',self.source,'-vf','setpts=if(lt(N\\,6)\\,N/(24*TB)\\,(6+2*(N-6))/(24*TB))','-fps_mode','vfr','-an',target])
        with self.assertRaisesRegex(ValueError,'Variable-frame-rate'):
            self.job(source=target)

if __name__=='__main__':
    unittest.main(verbosity=2)
