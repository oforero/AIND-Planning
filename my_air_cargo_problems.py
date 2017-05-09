from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # Create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            def build_load_action(cargo, plane, airport):
                act = expr("Load({}, {}, {})".format(cargo, plane, airport))
                precond_pos = [#expr("Cargo({})".format(cargo)),
                               #expr("Plane({})".format(plane)),
                               #expr("Airport({})".format(airport)),
                               expr("At({}, {})".format(cargo, airport)),
                               expr("At({}, {})".format(plane, airport))]
                precond_neg = []
                effect_add = [expr("In({}, {})".format(cargo, plane))]
                effect_rem = [expr("At({}, {})".format(cargo, airport))]

                action = Action(act, [precond_pos, precond_neg], [effect_add, effect_rem])
                return action

            loads = [build_load_action(cargo, plane, airport)
                     for cargo in self.cargos
                     for plane in self.planes
                     for airport in self.airports]
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            def build_unload_action(cargo, plane, airport):
                act = expr("Unload({}, {}, {})".format(cargo, plane, airport))
                precond_pos = [#expr("Cargo({})".format(cargo)),
                               #expr("Plane({})".format(plane)),
                               #expr("Airport({})".format(airport)),
                               expr("In({}, {})".format(cargo, plane)),
                               expr("At({}, {})".format(plane, airport))]
                precond_neg = []
                effect_add = [expr("At({}, {})".format(cargo, airport))]
                effect_rem = [expr("In({}, {})".format(cargo, plane))]

                action = Action(act, [precond_pos, precond_neg], [effect_add, effect_rem])
                return action

            unloads = [build_unload_action(cargo, plane, airport)
                       for cargo in self.cargos
                       for plane in self.planes
                       for airport in self.airports]

            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            def build_fly_action(plane, from_a, to_a):
                act = expr("Fly({}, {}, {})".format(plane, from_a, to_a))
                precond_pos = [#expr("Plane({})".format(plane)),
                               #expr("Airport({})".format(from_a)),
                               #expr("Airport({})".format(to_a)),
                               expr("At({}, {})".format(plane, from_a))]
                precond_neg = []
                effect_add = [expr("At({}, {})".format(plane, to_a))]
                effect_rem = [expr("At({}, {})".format(plane, from_a))]

                action = Action(act, [precond_pos, precond_neg], [effect_add, effect_rem])
                return action

            flights = [build_fly_action(plane, from_a, to_a)
                       for plane in self.planes
                       for from_a in self.airports
                       for to_a in self.airports
                       if from_a != to_a]

            return flights

        return load_actions() + fly_actions() + unload_actions()

    def print(self, state: str):
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        print("State: ", kb.clauses)

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        def all_positive(action, clauses):
            return all(map(lambda act: act in clauses, action.precond_pos))

        def none_negative(action, clauses):
            return all(map(lambda act: act not in clauses, action.precond_neg))

        kb = PropKB()
        state = decode_state(state, self.state_map)
        kb.tell(state.pos_sentence())

        possible_actions = [action for action in self.actions_list
                            if all_positive(action, kb.clauses) and none_negative(action, kb.clauses)]

        #print("State: ", state.pos, state.neg)
        #print("Here ", possible_actions)
        #if not possible_actions:
        #    print("No possible action: ", kb.clauses)
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        fluent = decode_state(state, self.state_map)
        pos = set(fluent.pos)
        neg = set(fluent.neg)

        #print("Old: ", fluent.pos, fluent.neg)
        #print("Action: ", action)

        for eff in action.effect_add:
            pos.add(eff)
            if eff in neg:
                neg.remove(eff)

        for eff in action.effect_rem:
            neg.add(eff)
            if eff in pos:
                pos.remove(eff)

        new_state = FluentState(list(pos), list(neg))
        #print("State Map: ", self.state_map)
        #print("Old state: ", sorted(fluent.pos), sorted(fluent.neg))
        #print("Action: ", action, action.effect_add, action.effect_rem)
        #print("New state: ", sorted(new_state.pos), sorted(new_state.neg))
        #print("New state: ", len(fluent.pos) + len(fluent.neg), len(new_state.pos) + len(new_state.neg))
        assert(len(fluent.pos) + len(fluent.neg) == len(new_state.pos) + len(new_state.neg))
        #print("New: ", new_state.pos, new_state.neg)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        #print("Testing: ", kb.clauses)

        return all(map(lambda clause: clause in kb.clauses, self.goal))

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    cargos_c = list(map(lambda x: expr("Cargo({})".format(x)), cargos))
    planes = ['P1', 'P2']
    planes_c = list(map(lambda x: expr("Plane({})".format(x)), planes))
    airports = ['JFK', 'SFO']
    airports_c = list(map(lambda x: expr("Airport({})".format(x)), airports))
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ] #+ cargos_c + planes_c + airports_c
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    cargos_c = list(map(lambda x: expr("Cargo({})".format(x)), cargos))
    planes = ['P1', 'P2', 'P3']
    planes_c = list(map(lambda x: expr("Plane({})".format(x)), planes))
    airports = ['JFK', 'SFO', 'ATL']
    airports_c = list(map(lambda x: expr("Airport({})".format(x)), airports))
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ] #+ cargos_c + planes_c + airports_c
    neg = [expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),
           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),
           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),

           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P3, JFK)'),
           expr('At(P3, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    cargos_c = list(map(lambda x: expr("Cargo({})".format(x)), cargos))
    planes = ['P1', 'P2']
    planes_c = list(map(lambda x: expr("Plane({})".format(x)), planes))
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    airports_c = list(map(lambda x: expr("Airport({})".format(x)), airports))
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),

           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),

           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),
           expr('At(C3, ORD)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),

           expr('At(C4, JFK)'),
           expr('At(C4, SFO)'),
           expr('At(C4, ATL)'),
           expr('In(C4, P1)'),
           expr('In(C4, P2)'),

           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
