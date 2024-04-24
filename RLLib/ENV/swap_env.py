import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from localcider.sequenceParameters import SequenceParameters
from sklearn.preprocessing import OneHotEncoder

class PeptDes_SWAP(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, env_config):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV")).reshape(-1, 1)
        self.enc = OneHotEncoder(handle_unknown='ignore').fit(self.alphabet)
        self.current_seq = list('MASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF')
        self.L = int(len(self.current_seq))
        self.target = 0.75
        
        self.action_space = spaces.MultiDiscrete(np.array([self.L,self.L]))
        self.observation_space = spaces.MultiBinary([self.L, 20])
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.L, 20), dtype=np.float32)
        #self.observation_space = spaces.Text(min_length=self.L, max_length=self.L, charset='ARNDCQEGHILKMFPSTWYV', seed=13)
        #self.observation_space = spaces.MultiDiscrete([2]*20*self.L)
    
    def logger(self, message):
        out = open(f'worker_{threading.get_ident()}.dat','a+')
        out.write(message)
        out.close()

    def fwd_onehot(self, seq):
        return self.enc.transform(np.array(seq).reshape(-1, 1)).toarray()
    def score_prot(self):
        SeqOb = SequenceParameters(''.join(self.current_seq))
        return SeqOb.get_kappa()

    def reset(self, seed=None, options=None):
        # Convert the protein sequence to one-hot encoding
        observation = self.fwd_onehot(self.current_seq)

        info = {}

        return observation, info

    def step(self, action):
        self.current_seq[action[1]], self.current_seq[action[0]] = self.current_seq[action[0]], self.current_seq[action[1]]
        # Calculate the new score
        score = self.score_prot()
        s = ''.join(self.current_seq)
        self.logger(message=f'{s} {action[0]} {action[1]} {score}\n')

      	# Determine the termination flag
        terminated = True

        reward = -abs(score - self.target)

      	# Prepare the new observation
        new_observation = self.fwd_onehot(self.current_seq)

        info = {}

        return new_observation, reward, terminated, False, info
