#!/usr/bin/env python3
"""Safely migrate locally generated, non-code DDS artifacts to SV naming."""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # reported per NumPy artifact; text/JSON migration remains usable
    np = None

RULES = (
    ("Distributional Direction Score", "Structural Variation"),
    ("Difficulty Direction Score", "Structural Variation"),
    ("Difficulty Direction", "Structural Variation"),
    ("DifficultyDirection", "StructuralVariation"),
    ("Difficulty_Direction", "Structural_Variation"),
    ("difficulty_direction", "structural_variation"),
    ("DDSResult", "SVResult"),
)
TOKEN_RULES = ((re.compile(r"(?<![A-Za-z0-9])DDS(?=$|[^A-Za-z0-9])"), "SV"),
               (re.compile(r"(?<![A-Za-z0-9])dds(?=$|[^A-Za-z0-9])"), "sv"))
TEXT_SUFFIXES = {'.txt','.csv','.tsv','.md','.rst','.tex','.yaml','.yml','.toml','.ini','.cfg','.log','.out','.xml','.html'}
JSON_SUFFIXES = {'.json','.jsonl','.ipynb'}
TORCH_SUFFIXES = {'.pt','.pth','.ckpt','.bin'}
PICKLE_SUFFIXES = {'.pkl','.pickle','.joblib'}
IMAGE_SUFFIXES = {'.png','.jpg','.jpeg','.gif','.bmp','.tif','.tiff','.webp','.svg'}
CODE_SUFFIXES = {'.py','.pyi','.pyc','.c','.cc','.cpp','.h','.hpp','.java','.js','.jsx','.ts','.tsx','.sh'}
EXCLUDED_NAMES = {'.git','.venv','venv','env','__pycache__','node_modules','site-packages','data','pretrained_clip'}

@dataclass
class Stats:
    scanned:int=0; modified:int=0; renamed_files:int=0; renamed_dirs:int=0
    json_modified:int=0; npz_modified:int=0; npy_modified:int=0; torch_modified:int=0; pickle_modified:int=0
    unchanged:int=0; conflicts:int=0
    unsupported:list[str]=field(default_factory=list); images:list[str]=field(default_factory=list); remaining:list[str]=field(default_factory=list)

def rename_text(value:str)->str:
    for old,new in RULES: value=value.replace(old,new)
    for pattern,new in TOKEN_RULES: value=pattern.sub(new,value)
    return value

def transform(obj:Any)->tuple[Any,bool]:
    if isinstance(obj,str):
        new=rename_text(obj); return new,new!=obj
    if isinstance(obj,dict):
        out={}; changed=False
        for key,value in obj.items():
            new_key=rename_text(key) if isinstance(key,str) else key
            new_value,value_changed=transform(value)
            if new_key in out:
                if values_equal(out[new_key],new_value): changed=True; continue
                raise ValueError(f"key collision after migration: {key!r} -> {new_key!r}")
            out[new_key]=new_value; changed |= new_key!=key or value_changed
        return out,changed
    if isinstance(obj,list):
        vals=[]; changed=False
        for value in obj:
            new,c=transform(value); vals.append(new); changed|=c
        return vals,changed
    if isinstance(obj,tuple):
        vals=[]; changed=False
        for value in obj:
            new,c=transform(value); vals.append(new); changed|=c
        return tuple(vals),changed
    if np is not None and isinstance(obj,np.ndarray) and obj.dtype.kind in 'UOS':
        if obj.dtype.kind=='O':
            flat=[]; changed=False
            for value in obj.flat:
                new,c=transform(value); flat.append(new); changed|=c
            return np.asarray(flat,dtype=object).reshape(obj.shape),changed
        flat=[rename_text(str(x)) for x in obj.flat]
        changed=any(a!=str(b) for a,b in zip(flat,obj.flat))
        return np.asarray(flat).reshape(obj.shape),changed
    return obj,False

def values_equal(a:Any,b:Any)->bool:
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray): return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True)
    try:return a==b
    except Exception:return False

def atomic_writer(path:Path, writer)->None:
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent); os.close(fd)
    tmp_path=Path(tmp)
    try: writer(tmp_path); os.replace(tmp_path,path)
    finally: tmp_path.unlink(missing_ok=True)

def process_json(path:Path,apply:bool)->bool:
    if path.suffix.lower()=='.jsonl':
        lines=[]; changed=False
        for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
            ending='\n' if line.endswith('\n') else ''; obj=json.loads(line)
            new,c=transform(obj); changed|=c; lines.append(json.dumps(new,ensure_ascii=False)+ending)
        if changed and apply: atomic_writer(path,lambda p:p.write_text(''.join(lines),encoding='utf-8'))
        return changed
    obj=json.loads(path.read_text(encoding='utf-8')); new,changed=transform(obj)
    if changed and apply: atomic_writer(path,lambda p:p.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'))
    return changed

def process_npz(path:Path,apply:bool)->bool:
    if np is None: raise RuntimeError('NumPy is required to migrate NPZ safely')
    with np.load(path,allow_pickle=True) as loaded:
        old={k:loaded[k] for k in loaded.files}
    new={}; changed=False
    for key,value in old.items():
        new_key=rename_text(key); new_value,c=transform(value)
        if key=='meta' and isinstance(new_value,np.ndarray) and new_value.shape==() and new_value.dtype.kind in 'US':
            try:
                meta=json.loads(str(new_value.item())); meta,c2=transform(meta)
                new_value=np.asarray(json.dumps(meta,ensure_ascii=False)); c|=c2
            except (json.JSONDecodeError,TypeError): pass
        if new_key in new and not values_equal(new[new_key],new_value): raise ValueError(f'NPZ key collision: {new_key}')
        new[new_key]=new_value; changed|=c or new_key!=key
    if changed and apply:
        def write(tmp):
            with open(tmp,'wb') as stream: np.savez_compressed(stream,**new)
            with np.load(tmp,allow_pickle=True) as check:
                for old_key,old_value in old.items():
                    nk=rename_text(old_key); nv=check[nk]
                    if old_value.dtype.kind not in 'UOS' and not values_equal(old_value,nv): raise ValueError(f'numeric array changed: {old_key}')
        atomic_writer(path,write)
    return changed

def process_npy(path:Path,apply:bool)->bool:
    if np is None: raise RuntimeError('NumPy is required to migrate NPY safely')
    old=np.load(path,allow_pickle=True); new,changed=transform(old)
    if changed and apply:
        def write(tmp):
            with open(tmp,'wb') as stream: np.save(stream,new)
            check=np.load(tmp,allow_pickle=True)
            if old.dtype.kind not in 'UOS' and not values_equal(old,check): raise ValueError('numeric NPY changed')
        atomic_writer(path,write)
    return changed

def tensor_snapshot(obj:Any,prefix='root')->dict[str,Any]:
    result={}
    try:
        import torch
        if torch.is_tensor(obj): result[prefix]=obj.detach().cpu().clone(); return result
    except ImportError:return result
    if isinstance(obj,dict):
        for k,v in obj.items(): result.update(tensor_snapshot(v,f'{prefix}/{k}'))
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj): result.update(tensor_snapshot(v,f'{prefix}/{i}'))
    return result

def process_torch(path:Path,apply:bool)->bool:
    import torch
    try: obj=torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: obj=torch.load(path,map_location='cpu')
    before=tensor_snapshot(obj); new,changed=transform(obj)
    if changed and apply:
        def write(tmp):
            torch.save(new,tmp)
            try: check=torch.load(tmp,map_location='cpu',weights_only=False)
            except TypeError: check=torch.load(tmp,map_location='cpu')
            after=tensor_snapshot(check)
            if before.keys()!=after.keys() or any(a.shape!=after[k].shape or a.dtype!=after[k].dtype or not torch.equal(a,after[k]) for k,a in before.items()): raise ValueError('Tensor changed')
        atomic_writer(path,write)
    return changed

def process_pickle(path:Path,apply:bool)->bool:
    loader=pickle.load; dumper=lambda obj,stream:pickle.dump(obj,stream,protocol=pickle.HIGHEST_PROTOCOL)
    if path.suffix.lower()=='.joblib':
        import joblib
        obj=joblib.load(path); new,changed=transform(obj)
        if changed and apply: atomic_writer(path,lambda p:joblib.dump(new,p))
        return changed
    with path.open('rb') as stream:obj=loader(stream)
    new,changed=transform(obj)
    if changed and apply: atomic_writer(path,lambda p:dumper(new,p.open('wb')))
    return changed

def excluded(path:Path,root:Path)->bool:return any(part in EXCLUDED_NAMES for part in path.relative_to(root).parts)

def migrate(root:Path,apply:bool,verbose:bool)->Stats:
    stats=Stats(); paths=[p for p in root.rglob('*') if not excluded(p,root)]
    for path in [p for p in paths if p.is_file()]:
        stats.scanned+=1; suffix=path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            if rename_text(path.name)!=path.name or path.name!=rename_text(path.name): stats.images.append(str(path.relative_to(root)))
            stats.unchanged+=1; continue
        if suffix in CODE_SUFFIXES: stats.unchanged+=1; continue
        try:
            if suffix in JSON_SUFFIXES: changed=process_json(path,apply); kind='json'
            elif suffix in TEXT_SUFFIXES:
                text=path.read_text(encoding='utf-8'); new=rename_text(text); changed=new!=text; kind='text'
                if changed and apply: atomic_writer(path,lambda p:p.write_text(new,encoding='utf-8'))
            elif suffix=='.npz': changed=process_npz(path,apply); kind='npz'
            elif suffix=='.npy': changed=process_npy(path,apply); kind='npy'
            elif suffix in TORCH_SUFFIXES: changed=process_torch(path,apply); kind='torch'
            elif suffix in PICKLE_SUFFIXES: changed=process_pickle(path,apply); kind='pickle'
            else: stats.unsupported.append(str(path.relative_to(root))); continue
            if changed:
                stats.modified+=1; setattr(stats,f'{kind}_modified',getattr(stats,f'{kind}_modified',0)+1)
                if verbose: print(f"{'MODIFY' if apply else 'WOULD MODIFY'} {path.relative_to(root)}")
            else: stats.unchanged+=1
        except Exception as exc:
            stats.conflicts+=1; print(f'CONFLICT/UNREADABLE {path.relative_to(root)}: {exc}',file=sys.stderr)
    # Rename deepest paths after content migration. Images and excluded trees are never renamed.
    candidates=[p for p in paths if not excluded(p,root) and p.suffix.lower() not in IMAGE_SUFFIXES | CODE_SUFFIXES and rename_text(p.name)!=p.name]
    if np is None:
        binary_parents={parent for p in paths if p.is_file() and p.suffix.lower() in {'.npz','.npy'} for parent in (p,*p.parents)}
        candidates=[p for p in candidates if p not in binary_parents and p.suffix.lower() not in {'.npz','.npy'}]
    for old in sorted(candidates,key=lambda p:len(p.parts),reverse=True):
        if not old.exists(): continue
        target=old.with_name(rename_text(old.name))
        if target.exists():
            identical=old.is_file() and target.is_file() and filecmp.cmp(old,target,shallow=False)
            if apply and identical: old.unlink()
            elif not identical: stats.conflicts+=1; print(f'PATH CONFLICT {old} -> {target}',file=sys.stderr); continue
        elif apply: old.rename(target)
        if old.is_dir(): stats.renamed_dirs+=1
        else: stats.renamed_files+=1
        if verbose: print(f"{'RENAME' if apply else 'WOULD RENAME'} {old.relative_to(root)} -> {target.relative_to(root)}")
    for p in root.rglob('*'):
        if '.git' not in p.parts and rename_text(p.name)!=p.name: stats.remaining.append(str(p.relative_to(root)))
    return stats

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,default=Path.cwd())
    mode=parser.add_mutually_exclusive_group(); mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--apply',action='store_true')
    parser.add_argument('--verbose',action='store_true'); args=parser.parse_args()
    root=args.root.resolve(); stats=migrate(root,args.apply,args.verbose)
    print('\nMigration summary')
    for key in ('scanned','modified','renamed_files','renamed_dirs','json_modified','npz_modified','npy_modified','torch_modified','pickle_modified','unchanged','conflicts'):
        print(f'  {key}: {getattr(stats,key)}')
    print(f'  unsupported/read failures: {len(stats.unsupported)}'); [print(f'    {x}') for x in stats.unsupported]
    print(f'  skipped images: {len(stats.images)}'); [print(f'    {x}') for x in stats.images]
    print(f'  remaining legacy paths: {len(stats.remaining)}'); [print(f'    {x}') for x in stats.remaining]
    return 1 if stats.conflicts else 0
if __name__=='__main__':raise SystemExit(main())
