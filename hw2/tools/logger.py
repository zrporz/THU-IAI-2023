from pathlib import Path
from datetime import datetime


class Logger:

    @classmethod
    def log(cls, filename, content):
        with open(Path.cwd() / "logs" / f"{filename}.log", 'a+', encoding='utf-8') as f:
            f.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ") + content + '\n')