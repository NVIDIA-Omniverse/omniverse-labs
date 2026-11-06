#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch one bounded CC0 studio HDRI; preserve exact source/license/hash metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlparse

URL='https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_09_1k.hdr'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'studio_small_09_1k.hdr';temporary=target.with_suffix('.download')
    maximum=32*1024*1024
    used=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    if used+maximum>1024**3:raise RuntimeError('Asset cache would exceed1GiB budget')
    if not target.exists():
        request=urllib.request.Request(URL,headers={'User-Agent':'Blender-OVStage-test-zoo/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            if urlparse(response.url).scheme!='https' or urlparse(response.url).hostname!='dl.polyhaven.org':
                raise RuntimeError('Unexpected asset redirect')
            if int(response.headers.get('Content-Length','0'))>maximum:raise RuntimeError('Asset exceeds32MiB bound')
            count=0
            try:
                with temporary.open('wb') as output:
                    while chunk:=response.read(65536):
                        count+=len(chunk)
                        if count>maximum:raise RuntimeError('Asset stream exceeds32MiB bound')
                        output.write(chunk)
                temporary.replace(target)
            finally:
                if temporary.exists():temporary.unlink()
    raw=target.read_bytes()
    if not raw.startswith((b'#?RADIANCE',b'#?RGBE')):raise RuntimeError('Downloaded file is not Radiance HDR')
    record={'id':'studio_small_09_1k','url':URL,'asset_page':'https://polyhaven.com/a/studio_small_09',
            'license':'CC0-1.0','license_url':'https://polyhaven.com/license','bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest(),'path':str(target.resolve()),'resolution_tier':'1k',
            'purpose':'Real HDR illumination/reflection control; procedural scenes need no external models'}
    (args.output/'assets.json').write_text(json.dumps({'schema':1,'assets':[record]},indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
