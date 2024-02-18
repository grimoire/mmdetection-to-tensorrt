import glob
import os
import shutil
from distutils.cmd import Command

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

# from build import build

package_data = {}

plugins_user_options = [
    ('plugins', None, 'Build plugins'),
    ('cuda-dir=', None, 'Location of CUDA (if not default location)'),
    ('torch-dir=', None, 'Location of PyTorch (if not default location)'),
    (
        'trt-inc-dir=',
        None,
        'Location of TensorRT include files (if not default location)',
    ),
    ('trt-lib-dir=', None,
     'Location of TensorRT libraries (if not default location)'),
]


def initialize_plugins_options(cmd_obj):
    cmd_obj.plugins = False
    cmd_obj.cuda_dir = None
    cmd_obj.torch_dir = None
    cmd_obj.trt_inc_dir = None
    cmd_obj.trt_lib_dir = None


class DevelopCommand(develop):
    description = 'Builds the package and symlinks it into the PYTHONPATH'
    user_options = develop.user_options + plugins_user_options

    def initialize_options(self):
        develop.initialize_options(self)
        initialize_plugins_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        # run_plugins_compilation(self)
        develop.run(self)


class InstallCommand(install):
    description = 'Builds the package'
    user_options = install.user_options + plugins_user_options

    def initialize_options(self):
        install.initialize_options(self)
        initialize_plugins_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # run_plugins_compilation(self)
        install.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    PY_CLEAN_FILES = [
        './build',
        './dist',
        './__pycache__',
        './*.pyc',
        './*.tgz',
        './*.egg-info',
    ]
    description = 'Command to tidy up the project root'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(
                os.path.normpath(os.path.join(root_dir, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(root_dir):
                    # Die if path in CLEAN_FILES is
                    # absolute + outside this directory
                    raise ValueError('%s is not a path inside %s' %
                                     (path, root_dir))
                print('Removing %s' % os.path.relpath(path))
                shutil.rmtree(path)

        cmd_list = {
            'Removing generated protobuf cc files':
            "find . -name '*.pb.cc' -print0 | xargs -0 rm -f;",
            'Removing generated protobuf h files':
            "find . -name '*.pb.h' -print0 | xargs -0 rm -f;",
            'Removing generated protobuf py files':
            "find . -name '*_pb2.py' -print0 | xargs -0 rm -f;",
            'Removing generated ninja files':
            "find . -name '*.ninja*' -print0 | xargs -0 rm -f;",
            'Removing generated o files':
            "find . -name '*.o' -print0 | xargs -0 rm -f;",
            'Removing generated so files':
            "find . -name '*.so' -print0 | xargs -0 rm -f;",
        }

        for cmd, script in cmd_list.items():
            print('{}'.format(cmd))
            os.system(script)


setup(
    name='mmdet2trt',
    version='0.6.0',
    author='q.yao',
    author_email='streetyao@live.com',
    description='mmdetection to tensorrt converter',
    url='https://github.com/grimoire/mmdetection-to-tensorrt',
    cmdclass={
        'install': InstallCommand,
        'clean': CleanCommand,
        'develop': DevelopCommand,
    },
    packages=find_packages(),
    package_data=package_data,
    entry_points={
        'console_scripts': [
            'mmdet2trt = mmdet2trt.mmdet2trt:main',
        ],
    },
)
