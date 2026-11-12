"""Recover the factory child-state evidence without sending controller input."""
import re


def factory_state_evidence(log_text):
    mode, state, event_stamp = None, None, None
    for line in log_text.splitlines():
        fields = line.split(None, 3)
        if len(fields) < 4 or fields[2] != '/sim2real_master_node':
            continue
        if 'ModeFsmDef::print' in line:
            match = re.search(r'\] from \w+ to (\w+)\s*$', line)
            if match:
                mode, state, event_stamp = match.group(1), None, fields[0]
        elif 'Sim2Real::execPdLoop' in line:
            match = re.search(r'\] State changed: \w+ -> (\w+)\s*$', line)
            if match and mode == 'ExecDefault':
                state, event_stamp = match.group(1), fields[0]
    return {'mode': mode, 'child_state': state, 'event_stamp': event_stamp}
