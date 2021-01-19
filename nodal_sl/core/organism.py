from .nodes import PredictionNode, SensoryNode, ActuatorNode


class Organism:

    def __init__(self, nodes):
        """predictions nodes should've been pre-structured"""
        for node in nodes:
            if isinstance(node, PredictionNode):
                node.build()
            node.reset_states()

        self.sensory_nodes = [node for node in nodes if isinstance(node, SensoryNode)]
        self.actuator_nodes = [node for node in nodes if isinstance(node, ActuatorNode)]
        self.prediction_nodes = [node for node in nodes if isinstance(node, PredictionNode)]

    def train(self):
        for prediction_node in self.prediction_nodes:
            prediction_node.train()

    def act(self, observations):
        """recieves and returns dict<str,any> structured observations and actions"""
        self._write_sensors(observations)
        self._thinking_step()
        return self._read_actuators()

    def _write_sensors(self, observations):
        for sensory_node in self.sensory_nodes:
            sensory_node.update_state(observations[sensory_node.name])

    def _read_actuators(self):
        return { actuator_node.name: actuator_node.get_action
                 for actuator_node in self.actuator_nodes }

    def _thinking_step(self):
        for prediction_node in self.prediction_nodes:
            prediction_node.bottom_up()
        for prediction_node in self.prediction_nodes:
            prediction_node.top_down()


def graph2brain(parent_relations,
                neighborhoods,
                predefined_nodes=None,
                prediction_node_kwargs=None):
    """Build connected node list for use in the `Organism`.

    :param parent_relations: parent-child relations to form in node network. Individual relations are
        given by `[parent,child]` string identifier lists. (`list<list<str>>`).
    :param neighborhoods: neighborhood group identifier lists (`list<list<str>>`).
    :param predefined_nodes: nodes that are already defined such as sensors, actuators, and manually
        constructed node graphs that this function will connect new nodes to. (`list<Node>`)
    :param prediction_node_kwargs: `dict` parameters to initialize `PredictionNode`'s with.
    :return: list of configured `Node` objects including origonal `Nodes` passed in
    """

    if predefined_nodes is None:
        predefined_nodes = list()
    assert isinstance(predefined_nodes, list)

    if prediction_node_kwargs is None:
        prediction_node_kwargs = dict()
    assert isinstance(prediction_node_kwargs, dict)

    nodes = predefined_nodes
    nodes = {node.name: node for node in nodes}

    all_parents = [parent for parent, _ in parent_relations]
    all_children = [child for _, child in parent_relations]
    all_neighbors = [item for sublist in neighborhoods for item in sublist]
    potential_new_identifiers = set(all_parents+all_children+all_neighbors)

    for id in potential_new_identifiers:
        if id not in nodes:
            nodes[id] = PredictionNode(name=id, **prediction_node_kwargs)

    for parent, child in zip(all_parents, all_children):
        nodes[child].add_parents([nodes[parent]])

    for neighborhood in neighborhoods:
        for node in neighborhood:
            nodes[node].add_neighbors([nodes[neighbor]
                                for neighbor in neighborhood])

    return list(nodes.values())