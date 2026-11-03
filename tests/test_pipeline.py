import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import backends
import pipeline
import prompt_builder as pb

BATCH = """
characters:
  ana: {description: woman with black hair, age: 30}
defaults:
  subjects: [ana]
  motion: she smiles
jobs:
  - {id: um, scene: beach}
  - {id: dois, scene: forest}
"""


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.images, self.videos = [], []

    def generate_image(self, prompt, width, height, seed, output_path, progress_cb=None):
        self.images.append(output_path.name)
        Image.new("RGB", (64, 64), (seed % 255, 10, 10)).save(output_path)
        return output_path

    def generate_video(self, prompt, image_path, width, height, seconds, seed, output_path, progress_cb=None):
        self.videos.append(output_path.name)
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.2",
                 "-pix_fmt", "yuv420p", str(output_path)],
                check=True,
            )
        else:
            output_path.write_bytes(b"fake")
        return output_path


class PipelineRunTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.backend = FakeBackend()
        patcher = mock.patch.object(backends, "make_backend", return_value=self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.batch_path = self.tmp / "lote.yaml"
        self.write(BATCH)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, body):
        self.batch_path.write_text(textwrap.dedent(body), encoding="utf-8")

    def run_batch(self, **kw):
        with mock.patch.object(pipeline, "log"):
            return pipeline.run_batch(pb.load_batch(self.batch_path), self.tmp / "out", **kw)

    def manifest(self):
        return json.loads((self.tmp / "out" / "lote" / "manifest.json").read_text())["jobs"]

    def test_full_run_stamps_and_resumes(self):
        summary = self.run_batch()
        self.assertEqual(summary["counts"], {"done": 2})
        self.assertEqual(len(self.backend.images), 2)
        self.assertEqual(len(self.backend.videos), 2)

        image = Path(self.manifest()["um"]["image_path"])
        with Image.open(image) as im:
            self.assertEqual(im.text["AI-Generated"], "true")
            self.assertEqual(json.loads(im.text["AI-Provenance"])["job_id"], "um")
        self.assertTrue(image.with_name(image.name + ".provenance.json").exists())

        if shutil.which("ffprobe"):
            video = self.manifest()["um"]["video_path"]
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
                                    "-of", "default=nw=1:nk=1", video], capture_output=True, text=True)
            self.assertIn("AI-generated", probe.stdout)

        # Retomar sem mudanças não gera nada de novo.
        self.run_batch()
        self.assertEqual(len(self.backend.images), 2)
        self.assertEqual(len(self.backend.videos), 2)

    def test_stop_after_images_then_curate(self):
        self.run_batch(stop_after="images")
        self.assertEqual(self.backend.videos, [])
        Path(self.manifest()["dois"]["image_path"]).unlink()  # curadoria: não gostei dessa

        summary = self.run_batch()
        self.assertEqual(summary["counts"], {"done": 1, "removed": 1})
        self.assertEqual(self.backend.videos, ["um.video.mp4"])

    def test_tampered_image_is_not_animated(self):
        self.run_batch(stop_after="images")
        Image.new("RGB", (64, 64), (0, 0, 255)).save(self.manifest()["um"]["image_path"])

        summary = self.run_batch()
        self.assertIn("alterada fora do pipeline", summary["failed"]["um"])
        self.assertEqual(self.backend.videos, ["dois.video.mp4"])

    def test_video_only_change_keeps_image(self):
        self.run_batch()
        self.write(BATCH.replace("motion: she smiles", "motion: she waves"))
        self.run_batch()
        self.assertEqual(len(self.backend.images), 2)
        self.assertEqual(len(self.backend.videos), 4)
        self.assertIn("She waves", self.manifest()["um"]["video_prompt"])

    def test_image_change_and_redo_regenerate(self):
        self.run_batch()
        self.write(BATCH.replace("scene: beach", "scene: desert"))
        self.run_batch(redo={"dois"})
        self.assertEqual(len(self.backend.images), 4)

    def test_rejected_job_never_reaches_backend(self):
        self.write(BATCH + "  - {id: tres, scene: celebrity lookalike}\n")
        summary = self.run_batch()
        self.assertIn("tres", summary["rejected"])
        self.assertNotIn("tres.image.png", self.backend.images)


class GpuLockTest(unittest.TestCase):
    def test_second_lock_fails_and_stale_lock_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".gpu.lock"
            with pipeline.GpuLock(path):
                with self.assertRaises(RuntimeError):
                    pipeline.GpuLock(path).__enter__()
            self.assertFalse(path.exists())
            path.write_text("999999999")  # pid que não existe
            with pipeline.GpuLock(path):
                self.assertEqual(path.read_text(), str(os.getpid()))


class ComfyHelpersTest(unittest.TestCase):
    def test_inject_inputs(self):
        wf = {"6": {"inputs": {"text": ""}}, "3": {"inputs": {"seed": 0}}}
        out = backends.inject_inputs(wf, {"prompt": ["6", "text"], "seed": ["3", "seed"]},
                                     {"prompt": "p", "seed": 5, "width": 9})
        self.assertEqual(out["6"]["inputs"]["text"], "p")
        self.assertEqual(out["3"]["inputs"]["seed"], 5)
        self.assertEqual(wf["6"]["inputs"]["text"], "")  # original intacto
        with self.assertRaises(backends.BackendError):
            backends.inject_inputs(wf, {"prompt": ["99", "text"]}, {"prompt": "p"})
        with self.assertRaises(backends.BackendError):
            backends.inject_inputs(wf, {}, {"prompt": "p"})

    def test_pick_output_prefers_saved_file(self):
        entry = {"outputs": {
            "9": {"images": [{"filename": "prev.png", "type": "temp"}]},
            "10": {"gifs": [{"filename": "final.mp4", "subfolder": "", "type": "output"}]},
            "11": {"images": [{"filename": "final.png", "type": "output"}]},
        }}
        self.assertEqual(backends.pick_output_file(entry, backends.VIDEO_EXTS)["filename"], "final.mp4")
        self.assertEqual(backends.pick_output_file(entry, backends.IMAGE_EXTS)["filename"], "final.png")
        with self.assertRaises(backends.BackendError):
            backends.pick_output_file({"outputs": {}}, backends.VIDEO_EXTS)

    def test_frames_grid(self):
        self.assertEqual(backends.frames_for(5, 24, 8), 121)  # LTX
        self.assertEqual(backends.frames_for(5, 16, 4), 81)   # Wan 2.2

    def test_ui_format_workflow_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = Path(tmp) / "ui.json"
            wf.write_text(json.dumps({"nodes": [], "links": []}))
            with self.assertRaises(backends.BackendError):
                backends.ComfyUIBackend({"image": {"workflow": str(wf)}, "video": {"workflow": str(wf)}}, Path(tmp))


if __name__ == "__main__":
    unittest.main()
