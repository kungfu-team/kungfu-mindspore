import os


def get_envs():
    for k in os.environ:
        yield (k)


for k in sorted(get_envs()):
    print(k)
