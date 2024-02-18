import os
from cog import BasePredictor, Input, Path
from typing import List
import sys, shutil, subprocess
sys.path.append('/content/LGM')
os.chdir('/content/LGM')

class Predictor(BasePredictor):
    def predict(
        self,
        ply_file: Path = Input(description="LGM .ply file"),
    ) -> Path:
        shutil.move(ply_file, '/content/test.ply')
        subprocess.run(['python', 'convert.py', 'big', '--force_cuda_rast', '--test_path', '/content/test.ply'])
        return Path('/content/test.glb')