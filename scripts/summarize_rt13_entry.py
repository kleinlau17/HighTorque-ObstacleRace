"""Summarize completed entry-fault runs using planner and guard traces."""
import argparse
import csv
import json
from pathlib import Path


def load_rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def summarize(path):
    with (path/'injection.json').open() as src:
        fault = json.load(src)
    with (path/'summary.json').open() as src:
        result = json.load(src)
    with (path/'frames.jsonl').open() as src:
        frames = list(map(json.loads,src))
    plans = load_rows(path/'decisions.csv')
    truth = load_rows(path/'truth_and_commands.csv')
    start, restore = fault['start_sim_s'], fault['restore_sim_s']
    until = restore if restore is not None else float('inf')

    def first_plan(key, from_time=0.):
        return next((float(row['t']) for row in plans
                     if row['reason']==key and float(row['t'])>=from_time), None)

    held = [row for row in frames if row.get('crate_entry_hold') or 'hold' in row['reason']]
    interval = ([row for row in truth if start <= float(row['t']) < until]
                if fault['mode'] != 'none' else [])
    # Report realized guard behavior over the injected interval, including the
    # transition to a hold. A 5 cm/s threshold is used only for this statistic.
    strafe = [row for row in interval if abs(float(row['forward_vy'])) >= .05]
    near_end = [row for row in interval if float(row['t']) >= float(interval[-1]['t'])-.5]
    latest = next((row for row in reversed(frames) if row['crate_error'] is not None), None)
    entry = first_plan('box_enter')
    along = first_plan('box_along')
    return dict(run=str(path),mode=fault['mode'],source=fault['source'],case=fault['case'],seed=fault['seed'],
        mirror=fault['mirror'],start_s=start,restore_s=restore,clock_end_s=result['time_s'],
        first_enter_s=entry,first_along_s=along,first_observed_hold_s=held[0]['t'] if held else None,
        hold_reasons=sorted({str(row.get('crate_entry_hold') or row['reason']) for row in held}),
        first_along_from_restore_s=first_plan('box_along',restore) if restore is not None else None,
        last_forwarded_strafe_during_fault_s=float(strafe[-1]['t']) if strafe else None,
        max_abs_forwarded_vy_in_last_half_second_of_fault_m_s=max((abs(float(row['forward_vy'])) for row in near_end),default=None),
        world_y_span_during_fault_m=max(float(x['y']) for x in interval)-min(float(x['y']) for x in interval) if interval else None,
        last_estimated_lateral_error_m=latest['crate_error'][1] if latest else None,
        masked_frames=fault['masked_frames'],frozen_calls=fault['frozen_command_calls'],
        frozen_calls_during_hold=fault.get('frozen_calls_during_hold'),
        result=result['result'],passed_boxes=[box['id'] for box in result['passed_boxes']],
        referee=result['referee'],contact=result['contact_detail'],last_reason=result['last_reason'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path,help='individual completed run directories')
    parser.add_argument('--out',type=Path,help='save the JSON summary to this file')
    args = parser.parse_args()
    payload = json.dumps([summarize(path) for path in args.runs],indent=2,ensure_ascii=False)+'\n'
    if args.out:
        args.out.write_text(payload)
    else:
        print(payload,end='')
