from core import Vision2DNode, PredictionNode, Organism

import gym

IMG_SIZE = (256,256)



eye = Vision2DNode(
    feature_depth=128,
    image_size=IMG_SIZE,
    pupil_size=(16,16),
    max_pupil_zoom=3
)

lower_neurons = []
for i in range(3):
    neuron = PredictionNode(
        f_abs=None,
        f_act=None,
        f_pred=None,
        d_zc=64,
        predictive_coding=True,
        name=f'lower_neuron{i}'
    )
    neuron.set_parents([eye])
    lower_neurons.append(neuron)

for neuron in lower_neurons:
    neuron.set_neighbors(lower_neurons)
    neuron.build()
    neuron.reset_states()

higher_neuron = PredictionNode(
    f_abs=None,
    f_act=None,
    f_pred=None,
    d_zc=128,
    predictive_coding=True,
    name='higher_neuron'
)
higher_neuron.set_parents(lower_neurons)
higher_neuron.build()

organism = Organism(nodes=[eye]+lower_neurons+[higher_neuron])

env = gym.make('MontezumaRevenge-v0')
for episode_i in range(10):
    obs = env.reset()
    for t in range(200):
        env.render()
        act = organism.act(obs)
        obs, r, done, info = env.step(act)
        if done:
            print('done')
            break
env.close()
