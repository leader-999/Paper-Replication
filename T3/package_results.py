"""Bundle only final-manifest runs, not speed-calibration or superseded runs."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    out=ROOT/'results';manifest=json.loads((out/'manifest.json').read_text())
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.txt','.cjs'] and p.name!='paper_text.txt']
    files.extend(p for p in out.rglob('*') if p.is_file() and p.name!='checksums.json' and 'runs' not in p.relative_to(out).parts)
    for case in manifest:files.extend(p for p in (ROOT/case['path']).rglob('*') if p.is_file())
    files=sorted(set(files));checksums={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checksumfile=out/'checksums.json';checksumfile.write_text(json.dumps(checksums,indent=2,ensure_ascii=False),encoding='utf-8')
    files.append(checksumfile)
    dest=ROOT.parent/'T3_独立复现_代码与结果.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in files:z.write(file,str(Path('T3复现')/file.relative_to(ROOT)))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print(dest,round(dest.stat().st_size/1024**2,2),'MiB',len(files),'files')
