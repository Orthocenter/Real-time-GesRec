import os

prev = [33, 26, 23, 29]
curr = [i for i in range(56, 59 + 1)]

for i in range(len(prev)):
    p = prev[i]
    c = curr[i]

    os.system('cp ../../expr/prepare_ems19.%02d.py prepare_ems19.%02d.py' % (p, c))
    os.system("sed -i '' 's/19.%02d/19.%02d/' prepare_ems19.%02d.py" % (p, c, c))

    os.system('cp ../../expr/ems19.%02d.sh ems19.%02d.sh' % (p, c))
    os.system("sed -i '' 's/CUDA_VISIBLE_DEVICES=0/CUDA_VISIBLE_DEVICES=1/' ems19.%02d.sh" % (c))
    os.system("sed -i '' 's/19.%02d/19.%02d/' ems19.%02d.sh" % (p, c, c))
    os.system("sed -i '' 's/batch_size 48/batch_size 12/' ems19.%02d.sh" % (c))