from ..core.special_nodes import Vision2DNode, DiscreetActuator
from ..core.organism import Organism, graph2brain


import gym

env = gym.make('MontezumaRevenge-v0')

IMG_SIZE = env.observation_space.shape[0:2]
left_eye = Vision2DNode(
    feature_depth=128,
    image_size=IMG_SIZE,
    pupil_size=(32, 32),
    max_pupil_zoom=3,
    name='left_eye'
)
right_eye = Vision2DNode(
    feature_depth=128,
    image_size=IMG_SIZE,
    pupil_size=(16,16),
    max_pupil_zoom=3,
    name='right_eye'
)
atari_controller = DiscreetActuator(n=env.action_space.n,
                                    name='atari_controller')
organism = Organism(nodes=graph2brain(
    parent_relations=[
        ['right_eye', 'node0a'],
        ['right_eye', 'node0b'],
        ['left_eye', 'node0b'],
        ['left_eye', 'node0c'],
        ['atari_controller', 'node1'],
        ['atari_controller', 'node0b'],
        ['node0a', 'node1'],
        ['node0b', 'node1'],
        ['node0c', 'node1'],
        ['node1', 'node2a'],
        ['node1', 'node2b'],
        ['node2a', 'node2b'],
        ['node2b', 'node2a'],],
    neighborhoods=[
        ['node0a', 'node0b', 'node0c'],
        ['node0b', 'node1']
    ],
    predefined_nodes=[left_eye, right_eye, atari_controller]
))

for episode_i in range(2):
    obs = env.reset()
    for t in range(100):
        env.render()
        act = organism.act(
            dict(left_eye=obs,right_eye=obs)
        )['atari_controller']
        obs, r, done, info = env.step(act)
        if done:
            print('done')
            break

env.close()
