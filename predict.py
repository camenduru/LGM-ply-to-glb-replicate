import os
from cog import BasePredictor, Input, Path
from typing import List
import sys, shutil, subprocess
sys.path.append('/content/LGM')
os.chdir('/content/LGM')
current_path = os.environ.get('PATH', '')
print(current_path)
new_path = current_path + os.pathsep + '/usr/local/cuda/bin'
os.environ['PATH'] = new_path
print(new_path)

class Predictor(BasePredictor):
    def predict(
        self,
        ply_file: Path = Input(description="LGM .ply file"),
    ) -> Path:
        shutil.move(ply_file, '/content/test.ply')
        subprocess.run(['python', 'convert.py', 'big', '--force_cuda_rast', '--test_path', '/content/test.ply'])
        return Path('/content/test.glb')