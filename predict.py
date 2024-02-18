import os
from cog import BasePredictor, Input, Path
from typing import List
import sys, subprocess, requests
sys.path.append('/content/LGM')
os.chdir('/content/LGM')
current_path = os.environ.get('PATH', '')
print(current_path)
new_path = current_path + os.pathsep + '/usr/local/cuda/bin'
os.environ['PATH'] = new_path
print(new_path)

class Predictor(BasePredictor):
    def setup(self) -> None:
        response = requests.get('https://replicate.delivery/pbxt/UvKKgNj9mT7pIVHzwerhcjkp5cMH4FS5emPVghk2qyzMRwUSA/gradio_output.ply')
        response.raise_for_status()
        with open('/content/test.ply', 'wb') as f:
            f.write(response.content)
        subprocess.run(['python', 'convert.py', 'big', '--force_cuda_rast', '--test_path', '/content/test.ply'])
    def predict(
        self,
        ply_file_url: str = Input(description="URL of LGM .ply file"),
    ) -> Path:
        response = requests.get(ply_file_url)
        response.raise_for_status()
        with open('/content/test.ply', 'wb') as f:
            f.write(response.content)
        subprocess.run(['python', 'convert.py', 'big', '--force_cuda_rast', '--test_path', '/content/test.ply'])
        return Path('/content/test.glb')