"""Universal adapter: implement decide() in agent_logic.py for your own Agent."""
import os
import time
from third_party_agent import browser
from agent_logic import decide


class MyAgent:
    def reset(self, request):
        self.run_id = request["run_id"]
        self.state = {}

    def step(self, request):
        if request["run_id"] != self.run_id:
            raise ValueError("run_id mismatch; reset is required")
        deadline = request.get("execution_deadline")
        self.state["_remaining_seconds"] = max(0, deadline-time.time()) if deadline else None
        self.state.pop('_screenshot_path', None)
        directory = os.getenv('WEBARENA_SCREENSHOT_DIR')
        if directory:
            path = browser.save_screenshot(request['observation'], directory)
            if path is not None:
                self.state['_screenshot_path'] = str(path)
        result = decide(
            task=request["task"],
            observation=request["observation"],
            history=request.get("history", []),
            action_history=request.get("action_history", []),
            state=self.state,
        )
        action = result if isinstance(result, str) else result.get("action") if isinstance(result, dict) else None
        if not isinstance(action, str) or not action.strip():
            raise ValueError('decide must return an action string or {"action": "..."}')
        return {"action": action.strip()}

    def close(self):
        self.state = {}
