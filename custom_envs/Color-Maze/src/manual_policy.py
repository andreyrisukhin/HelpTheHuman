from pynput import keyboard
import time


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]

        self.show_obs = show_obs

        # action mappings for all agents are the same
        self.action_mapping = dict()
        self.default_action = 0
        if agent_id == 0:
            self.action_mapping['w'] = 0  # up
            self.action_mapping['s'] = 1  # down
            self.action_mapping['a'] = 2  # left
            self.action_mapping['d'] = 3  # right
        elif agent_id == 1:
            self.action_mapping = dict()
            self.action_mapping[keyboard.Key.up] = 0  # up
            self.action_mapping[keyboard.Key.down] = 1  # down
            self.action_mapping[keyboard.Key.left] = 2  # left
            self.action_mapping[keyboard.Key.right] = 3  # right
        else:
            assert False

        self.current_action = None

    def on_key_press(self, key):
        if isinstance(key, keyboard.KeyCode):
            key = key.char
        elif isinstance(key, keyboard.Key):
            key = key
        else:
            assert False

        if key in self.action_mapping:
            self.current_action = self.action_mapping[key]

    def __call__(self, observation, agent):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        # set the default action
        action = self.default_action

        print(f'Agent {agent} input:\n')
        # if we get a key, override action using the dict
        with keyboard.Events() as events:
            event = events.get()
            if event:
                self.on_key_press(event.key)
                time.sleep(0.2)


        if self.current_action:
            action = self.current_action
            self.current_action = None

        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping
