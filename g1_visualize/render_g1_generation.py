#!/usr/bin/env python3
"""Render every native trajectory in a gen_g1 output directory."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_visualize.replay_g1 import DEFAULT_MJCF, qpos_addresses, render_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-dir', type=Path, required=True)
    parser.add_argument('--mjcf', type=Path, default=DEFAULT_MJCF)
    args = parser.parse_args()
    root = args.generation_dir
    manifest = json.loads((root / 'generation_manifest.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    root_address, joint_addresses = qpos_addresses(model)
    lines = ['# Text-to-motion samples', '',
             'Pure text-conditioned generation, not RVQ reconstruction. Two stochastic samples per caption.',
             'Kinematic replay only; the scene does not contain a box or simulate contacts.', '',
             '| Sample | Caption | Repeat 0 | Repeat 1 |', '| --- | --- | --- | --- |']
    contact = Image.new('RGB', (960, len(manifest['prompts']) * 265), 'white')
    draw = ImageDraw.Draw(contact)
    for i, (caption, length) in enumerate(zip(manifest['prompts'], manifest['lengths'])):
        videos = []
        for repeat in range(manifest['repeats']):
            stem = f'sample{i:03d}_repeat{repeat:02d}_len{length}'
            native = np.load(root / 'motions' / f'{stem}_native38.npy')
            if native.shape != (length, 38) or not np.isfinite(native).all():
                raise ValueError(f'Invalid generated trajectory: {stem}')
            output = root / 'videos' / f'{stem}.mp4'
            render_args = SimpleNamespace(output_mp4=output, width=640, height=480,
                camera_azimuth=50., camera_elevation=-15., camera_distance=3.,
                orbit_degrees=0., clip_joint_limits=False)
            render_video(model, data, native, manifest['model_fps'], manifest['model_fps'],
                         root_address, joint_addresses, render_args)
            with imageio.get_reader(output) as reader:
                if reader.count_frames() != length:
                    raise ValueError(f'Incorrect video length: {output}')
                frames = [reader.get_data(t) for t in (0, length // 2, length - 1)]
                if max(float(np.std(f)) for f in frames) < 2:
                    raise ValueError(f'Blank video: {output}')
                if repeat == 0:
                    for col, frame in enumerate(frames):
                        contact.paste(Image.fromarray(frame).resize((320, 240)), (col * 320, i * 265 + 25))
            videos.append(f'[video {repeat}](videos/{stem}.mp4)')
        lines.append(f'| {i} | {caption} | ' + ' | '.join(videos) + ' |')
        draw.text((5, i * 265 + 5), f'{i}: {caption[:130]}', fill='black')
    contact.save(root / 'preview.jpg')
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'INDEX={root / "README.md"}')


if __name__ == '__main__':
    main()
