"""Replace this class with your own agent. This demo does not solve tasks."""


class ExampleAgent:
    def reset(self, request):
        self.memory = []

    def step(self, request):
        # request: task, observation (url/text), history, action_history, step.
        # Call your own model/planner here and return ONE WebArena action.
        return {"action": "stop [SDK connection verified; demo did not solve the task]"}

    def close(self):
        self.memory = []
