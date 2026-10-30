"""Public minimal integration example. Replace decide() with your own Agent."""
import json
from third_party_agent import model

SYSTEM = '''Use the current accessibility tree to perform the user's web task.
Return only JSON: {"action":"one WebArena action"}.
Examples: click [12], type [12] [text] [0], press [Tab], scroll [down],
goto [URL], go_back, go_forward, stop [answer].
Use element IDs from the current observation. Return one action per call.'''


def decide(*, task, observation, history, action_history, state):
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': json.dumps({
            'task': task, 'url': observation.get('url', ''),
            'page': observation.get('text', ''),
        }, ensure_ascii=False)},
    ]
    reply = model.generate(messages)
    result = json.loads(reply.text)
    if not isinstance(result, dict) or not isinstance(result.get('action'), str) or not result['action'].strip():
        raise ValueError('Expected JSON with a nonempty action string')
    return {'action': result['action'].strip()}
