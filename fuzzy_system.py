import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl


def main():
    gd, grade_system = fuzzy_system()
    distance = 120
    angle = 0
    grade(distance, angle, gd, grade_system)


def fuzzy_system():
    distance = ctrl.Antecedent(np.arange(-1, 141, 0.01), 'Distance')
    angle = ctrl.Antecedent(np.arange(-1, 180, 0.1), 'Angle')
    grade = ctrl.Consequent(np.arange(-1, 101, 0.01), 'Score')

    distance['Motionless'] = fuzz.trapmf(distance.universe, [0, 0, 10, 20])
    distance['Short'] = fuzz.trimf(distance.universe, [10, 40, 70])
    distance['Medium'] = fuzz.trimf(distance.universe, [40, 70, 100])
    distance['Long'] = fuzz.trapmf(distance.universe, [70, 100, 130, 130])

    angle['Small'] = fuzz.trapmf(angle.universe, [0, 0, 15, 30])
    angle['Medium'] = fuzz.trimf(angle.universe, [15, 30, 45])
    angle['Large'] = fuzz.trapmf(angle.universe, [30, 45, 180, 180])

    grade['Dead'] = fuzz.trapmf(grade.universe, [0, 0, 25, 45])
    grade['Bad'] = fuzz.trimf(grade.universe, [25, 45, 65])
    grade['Normal'] = fuzz.trimf(grade.universe, [45, 65, 85])
    grade['Good'] = fuzz.trapmf(grade.universe, [65, 85, 100, 100])

    # # distance
    # distance.view()
    # plt.show()
    # # angle
    # angle.view()
    # plt.show()
    # # grade
    # grade.view()
    # plt.show()

    # define fuzzy rule
    rule1 = ctrl.Rule(distance['Motionless'] & angle['Small'], grade['Dead'])
    rule2 = ctrl.Rule(distance['Motionless'] & angle['Medium'], grade['Dead'])
    rule3 = ctrl.Rule(distance['Motionless'] & angle['Large'], grade['Dead'])

    rule4 = ctrl.Rule(distance['Short'] & angle['Small'], grade['Bad'])
    rule5 = ctrl.Rule(distance['Short'] & angle['Medium'], grade['Bad'])
    rule6 = ctrl.Rule(distance['Short'] & angle['Large'], grade['Bad'])

    rule7 = ctrl.Rule(distance['Medium'] & angle['Small'], grade['Normal'])
    rule8 = ctrl.Rule(distance['Medium'] & angle['Medium'], grade['Bad'])
    rule9 = ctrl.Rule(distance['Medium'] & angle['Large'], grade['Bad'])

    rule10 = ctrl.Rule(distance['Long'] & angle['Small'], grade['Good'])
    rule11 = ctrl.Rule(distance['Long'] & angle['Medium'], grade['Normal'])
    rule12 = ctrl.Rule(distance['Long'] & angle['Large'], grade['Bad'])

    grade_ctrl = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
        rule11, rule12
    ])
    grade_system = ctrl.ControlSystemSimulation(grade_ctrl)

    return grade, grade_system


def grade(distance, angle, grade, grade_system):
    grade_system.input['Distance'] = distance
    grade_system.input['Angle'] = angle
    grade_system.compute()

    # grade.view(sim=grade_system)
    # plt.show()

    # print(grade_system.output['Score'])

    return grade_system.output['Score']


if __name__ == '__main__':
    main()