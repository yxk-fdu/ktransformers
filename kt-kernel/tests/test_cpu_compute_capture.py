import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for parts in [1,2]:
    with tempfile.TemporaryDirectory(prefix='cpu_compute_verify_', dir='/tmp') as tmp:
        env = dict(os.environ, KT_MOE_CPU_TRACE=tmp+'/compute_%p.jsonl', KT_MOE_PIN_WEIGHTS='1')
        subprocess.run([sys.executable, str(Path(__file__).with_name('test_strict_pin.py')), str(parts)], env=env, check=True)
        files = list(Path(tmp).glob('compute_*.jsonl'))
        assert len(files) == 1, files
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        spans = [r for r in records if r['event'] == 'cpu_moe.compute']
        assert len(spans) == parts * 2, spans
        assert all(r['expert_ids'] == [0] and r['expert_count'] == 1 for r in spans)
        assert all(r['tokens'] == 1 and r['duration_ns'] > 0 and r['wall_ns'] > 1700000000000000000 for r in spans)
        assert all(not r['expert_ids_truncated'] for r in spans)
        assert set(r['part'] for r in spans) == set(range(parts))
        assert records[-1]['event'] == 'cpu_moe.trace_flush' and records[-1]['dropped'] == 0
        print('PASS actual compute capture: parts=', parts, 'spans=', len(spans), 'no dropped samples')
