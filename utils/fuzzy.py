import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl


def main():
    motility_base, motility_ = fuzzy_system()
    moving_distance = 300
    path_curvature = 0.05
    motility_system(motility_base, motility_, moving_distance, path_curvature)


def fuzzy_system():
    distance = ctrl.Antecedent(np.arange(0, 320.01, 0.01), 'Moving Distance')
    curvature = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'Path Curvature')
    motility = ctrl.Consequent(np.arange(0, 100.01, 0.01), 'Motility')

    distance['Motionless'] = fuzz.trapmf(distance.universe, [0, 0, 30, 60])
    distance['Short'] = fuzz.trapmf(distance.universe, [30, 60, 90, 120])
    distance['Medium'] = fuzz.trapmf(distance.universe, [90, 120, 150, 180])
    distance['Long'] = fuzz.trapmf(distance.universe, [150, 180, 320, 320])

    curvature['Small'] = fuzz.trapmf(curvature.universe, [0, 0, 0.10, 0.20])
    curvature['Medium'] = fuzz.trapmf(curvature.universe,
                                      [0.10, 0.20, 0.30, 0.40])
    curvature['Large'] = fuzz.trapmf(curvature.universe, [0.30, 0.40, 1, 1])

    motility['IM'] = fuzz.trapmf(motility.universe, [0, 0, 20, 30])
    motility['NP'] = fuzz.trapmf(motility.universe, [20, 30, 50, 60])
    motility['PR'] = fuzz.trapmf(motility.universe, [50, 60, 100, 100])

    # # distance
    # distance.view()
    # plt.show()
    # # curvature
    # curvature.view()
    # plt.show()
    # # motility
    # motility.view()
    # plt.show()

    # define fuzzy rule
    rule1 = ctrl.Rule(distance['Motionless'] & curvature['Small'],
                      motility['IM'])
    rule2 = ctrl.Rule(distance['Motionless'] & curvature['Medium'],
                      motility['IM'])
    rule3 = ctrl.Rule(distance['Motionless'] & curvature['Large'],
                      motility['IM'])

    rule4 = ctrl.Rule(distance['Short'] & curvature['Small'], motility['NP'])
    rule5 = ctrl.Rule(distance['Short'] & curvature['Medium'], motility['NP'])
    rule6 = ctrl.Rule(distance['Short'] & curvature['Large'], motility['NP'])

    rule7 = ctrl.Rule(distance['Medium'] & curvature['Small'], motility['PR'])
    rule8 = ctrl.Rule(distance['Medium'] & curvature['Medium'], motility['PR'])
    rule9 = ctrl.Rule(distance['Medium'] & curvature['Large'], motility['NP'])

    rule10 = ctrl.Rule(distance['Long'] & curvature['Small'], motility['PR'])
    rule11 = ctrl.Rule(distance['Long'] & curvature['Medium'], motility['PR'])
    rule12 = ctrl.Rule(distance['Long'] & curvature['Large'], motility['NP'])

    motility_ctrl = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
        rule11, rule12
    ])
    motility_base = ctrl.ControlSystemSimulation(motility_ctrl)

    return motility_base, motility


def motility_system(motility_base, motility, moving_distance, path_curvature):
    motility_base.input['Moving Distance'] = moving_distance
    motility_base.input['Path Curvature'] = path_curvature
    motility_base.compute()

    # 輸出圖
    # motility.view(sim=motility_base)
    # plt.show()

    # 輸出分數
    # print(motility_base.output['Motility'])

    return motility_base.output['Motility']


if __name__ == '__main__':
    main()