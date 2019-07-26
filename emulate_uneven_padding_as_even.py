'''
Split a file while emulating uneven padding.
'''

import os
import glob
import sys
from subprocess import call



#### begin of config

filename = 'quick'

fps = 30
delay = 4/30
duration = 10 / 30

num_frames_per_gesture = {
    'quick_pronation': 10,
    'quick_supination': 7,
    'quick_wrist_left': 8,
    'quick_wrist_right': 9,
}

#### end of config

def split(working_dir, num_frames_per_gesture):
    print(num_frames_per_gesture)
    os.chdir(working_dir)

    path = 'rgb/%s.mov' % filename
    directory = path.split(".")[0] + "_all"
    if not os.path.exists(directory):
        os.makedirs(directory)
        call(["ffmpeg", "-i",  path, os.path.join(directory, "%05d.jpg"), "-hide_banner"])

    with open('%s.txt' % filename, 'r') as f:
        annot = f.readlines()

    annot = [a for a in annot[0::2]]
    ges_cnt = {}

    frame_indices = [i for i in range(int(delay * fps) + 1)]

    for a in annot[:]:
        ges = a.split('start')[0]
        ges = '_'.join(ges.lower().strip().split(' '))
        # ges = 'human_' + ges

        t = a.split('start:')[-1].strip()
        t = float(t)

        start = int((t + delay) * fps)
        end = start + num_frames_per_gesture[ges]
        for i in range(start, end):
            frame_indices.append(i)

    overall_cnt = 0
    for a in annot[:]:
        ges = a.split('start')[0]
        ges = '_'.join(ges.lower().strip().split(' '))
        # ges = 'human_' + ges

        t = a.split('start:')[-1].strip()
        t = float(t)

        start = int((t + delay) * fps)
        end = int((t + delay + duration) * fps)

        if end >= len(frame_indices):
            break

        cnt = ges_cnt.get(ges, 0) + 1
        overall_cnt += 1
        ges_cnt[ges] = cnt
        output_dir = 'rgb/{:04d}_{}_{:02d}_all'.format(overall_cnt, ges, cnt)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(start, end):
            os.system('cp {}/{:05d}.jpg {}'.format(directory, frame_indices[i], output_dir))

        d = sorted(glob.glob('%s/*' % output_dir))

        for i in range(len(d)):
            f = '%05d.jpg' % (i+1)
            d2 = output_dir + '/' + f
            os.system('mv %s %s' % (d[i], d2))

if __name__ == 'main':
    if len(sys.argv) <= 1:
        raise Exception()
    split(working_dir=sys.argv[1], num_frames_per_gesture=num_frames_per_gesture)