"""Check numbers transcribed into this report against the local evidence bundle."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
analysis = json.loads((root / 'evidence/p25_analysis.json').read_text(encoding='utf-8'))
sens = json.loads((root / 'evidence/descriptive_sensitivity.json').read_text(encoding='utf-8'))
expected = {'C_19': (69, 7226, 10472, 8, 627, 148),
            'D_19': (76, 7967, 10703, 10, 649, 163),
            'C_1': (65, 7326, 10589, 7, 629, 148),
            'D_1': (75, 7838, 10557, 12, 649, 162)}
for key, row in expected.items():
    run = analysis['runs'][key]
    a, f = run['audit'], run['fit']
    assert (a['dead_hit'], a['matched'], a['candidates'], a['small_hit'], f['dead_hit'], f['small_hit']) == row
    assert run['optimizer_steps'] == 17420
    assert a['dead_total'] == 145 and a['total'] == 12250 and f['dead_total'] == 656 and f['total'] == 50968
for seed, counts, lower, upper in [('19', (26, 11, 6, 10), .016, .13541666666666666), ('1', (30, 8, 4, 11), .032, .14583333333333334)]:
    x = sens['fixed_examples'][seed]['totals']
    assert tuple(x[k] for k in ('retained', 'gained', 'lost', 'missed_both')) == counts
    assert sum(counts) == x['Dead_objects'] == 53
    assert sens['leave_one_positive_image_out'][seed]['min'] == lower
    assert sens['leave_one_positive_image_out'][seed]['max'] == upper
    assert not analysis['pairs'][seed]['pass_seed']
lines = [s for s in analysis['sacct'].splitlines() if s.split('|')[0] in [f'55635726_{i}' for i in range(4)]]
seconds = sum(sum(int(part) * factor for part, factor in zip(s.split('|')[4].split(':'), (3600, 60, 1))) for s in lines)
assert len(lines) == 4 and seconds == 71957
print('P25 table, sensitivity, gates and top-level GPU seconds verified:', seconds)
