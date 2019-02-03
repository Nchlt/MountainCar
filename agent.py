'''
Nouredine Nour (LaChancla sur codalab)
TP4  Direct Policy Search
'''

import numpy as np
from environment import Environment
import cma
"""
Contains the definition of the agent that will run in an
environment.
"""

class Patrick:

    def __init__(self):
        """
        Init a new agent.
        """

        # Discretizing position and velocity :
        self.positions = np.linspace(start=-1.2, stop=0.6, num=6)
        self.velocities = np.linspace(start=-0.07, stop=0.07, num=20)

        # Here we initialize the policy with a previous solution
        self.policy = np.array([list([1., 0., 2., 2., 0., 2., 2., 2., 0., 2., 1., 1., 0., 2., 1., 1., 0.,
               1., 2., 2.]), list([2., 2., 2., 1., 2., 0., 1., 0., 0., 0., 2., 2., 1., 2., 2., 2., 1.,
               0., 2., 1.]), list([0., 2., 2., 0., 0., 0., 0., 1., 1., 2., 2., 2., 2., 1., 2., 0., 2.,
               0., 1., 2.]), list([0., 2., 0., 2., 1., 0., 0., 0., 1., 0., 1., 2., 2., 2., 2., 2., 2.,
               2., 0., 0.]), list([1., 2., 1., 1., 0., 2., 2., 2., 0., 2., 2., 0., 2., 2., 2., 2., 2.,
               2., 1., 1.]), list([0., 0., 2., 0., 0., 2., 2., 2., 2., 0., 2., 1., 0., 0., 0., 2., 1.,
               0., 1., 2.])])
        # We can start from scratch by un commenting the following line:
        # self.policy = np.zeros(shape=(self.positions.shape[0], self.velocities.shape[0]))
        # We define the initial solution for our cma-es
        self.init = self.policy.reshape(self.positions.shape[0]*self.velocities.shape[0])
        self.b_policy = None
        self.train() # Do not remove this line!!


    def train(self):

        """
        Learn your (final) policy.

        Use evolution strategy algortihm CMA-ES: https://pypi.org/project/cma/

        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        def policy_action(position, velocity, policy):
                '''Fonction that returns the action given a state and a policy'''
                i_position, i_velocity = get_discretized_env(position, velocity)
                # print(i_position, i_velocity)
                action = policy[i_position][i_velocity]
                return action

        def get_discretized_env(position, velocity, velocities=self.velocities, positions=self.positions):
                '''Fonction that give the indices to look for in the discretized position and velocity space'''
                i = 0
                while velocity > velocities[i]:
                    i += 1
                velocity_index = i
                j = 0
                while position > positions[j]:
                    j += 1
                position_index = j
                return position_index, velocity_index

        env = Environment()

        # For debug purposes
        self.min_value = 999999999

        def obj_function(policy):
            ''' Fonction that takes a policy and run it on 200 steps of the environement. It returns the fitness of the policy.
            '''
            env.reset()
            iter_counter = 0
            x = policy.reshape(self.positions.shape[0]*self.velocities.shape[0], -1)
            x = np.floor(x)
            d_policy = x.reshape(self.positions.shape[0], self.velocities.shape[0])

            distances = []
            distance_mid = []
            energy = []
            malus = 200

            for i in range(200):
                # env.render()
                # We take an action according to the given policy
                position, velocity = env.state
                distances.append(np.absolute(0.6 - position))
                distance_mid.append(np.absolute(-0.56 - position))
                energy.append(0.5*(velocity**2))
                if position == 0.6:
                    # If we enter here we won the game
                    malus = (i / 200) * 200
                    value = -sum(distance_mid) -max(energy) + malus + min(distances)*50 - (np.absolute(min(distances) - max(distances))*100)

                    # For debug purposes :
                    if value < self.min_value:
                        self.min_value = value
                        print('New best value = '+str(self.min_value))
                    return value
                action = policy_action(position, velocity, d_policy)
                _, _ = env.act(int(np.floor(action)))

            value = -sum(distance_mid)-max(energy) + malus + min(distances)*50 -(np.absolute(min(distances) - max(distances))*100)
            # For debug purposes :
            if value < self.min_value:
                self.min_value = value
                print('New best value = '+str(self.min_value))
            return value

        # We launch a cma-es to find a policy that minimizes the ojective function value
        # We decided to fix the ftarget value so it doest take too long too run but we could remove it
        # to optimize the function even more
        best_policy, _ = cma.fmin2(obj_function, self.init, 2,{
        #'BoundaryHandler': 'BoundPenalty',
        'BoundaryHandler': 'BoundTransform',
        'bounds':[0,3],
        'verbose':1,
        'ftarget':-100,
        'seed': 237591
        })
        print("Optimization FINISHED")
        #self.policy = best_policy
        self.b_policy = best_policy
        self.policy = np.floor(best_policy).reshape(self.positions.shape[0], self.velocities.shape[0])
        print("Best Policy updated"+str(self.policy))

    def act(self, observation):
        """
        Acts given an observation of the environment (using learned policy).

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """

        def policy_action(position, velocity, policy):
                i_position, i_velocity = get_discretized_env(position, velocity)
                # print(i_position, i_velocity)

                action = policy[i_position][i_velocity]
                # print("Velocity : %f"%velocity)
                # print("Next Action %d, %d ----> %d"%(i_position, i_velocity, action))
                return action

        def get_discretized_env(position, velocity, velocities=self.velocities, positions=self.positions):
                i = 0
                while velocity > velocities[i]:
                    i += 1
                velocity_index = i
                j = 0
                while position > positions[j]:
                    j += 1
                position_index = j
                return position_index, velocity_index

        # Once the training is finished we simply follow the best policy cma could find
        x = self.b_policy.reshape(self.positions.shape[0]*self.velocities.shape[0], -1)
        x = np.floor(x)
        d_policy = x.reshape(self.positions.shape[0], self.velocities.shape[0])

        action = policy_action(observation[0], observation[1], d_policy)
        action = (int(np.floor(action)))

        return action

Agent = Patrick
