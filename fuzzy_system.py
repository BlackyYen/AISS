import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

def main():
    gd, grade_system = fuzzy_system()
    distance = 60
    angle = 10
    grade(gd, grade_system, distance, angle)
    
#%%
def fuzzy_system():
    # New Antecedent/Consequent objects hold universe variables and membership
    # functions
    distance = ctrl.Antecedent(np.arange(0, 140.1, 0.1), 'Distance')
    angle = ctrl.Antecedent(np.arange(0, 180.1, 0.1), 'Angle')
    grade = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'Score')
    
# =============================================================================
#     distance['Motionless'] = fuzz.trapmf(distance.universe, [0, 0, 3 ,5])
#     distance['Short'] = fuzz.trimf(distance.universe, [3, 20, 35])
#     distance['Medium'] = fuzz.trimf(distance.universe, [20, 35, 50])
#     distance['Long'] = fuzz.trapmf(distance.universe, [35, 50, 75,  75])
# =============================================================================

    distance['Motionless'] = fuzz.trapmf(distance.universe, [0, 0, 4 ,8])
    distance['Short'] = fuzz.trimf(distance.universe, [5, 37.5, 70])
    distance['Medium'] = fuzz.trimf(distance.universe, [37.5, 68.75, 100])
    distance['Long'] = fuzz.trapmf(distance.universe, [68.75, 100, 140,  140])

    angle['Small'] = fuzz.trapmf(angle.universe, [0, 0, 15, 30])
    angle['Medium'] = fuzz.trimf(angle.universe, [15, 30, 45])
    angle['Large'] = fuzz.trapmf(angle.universe, [30, 45, 180, 180])
    
    grade['Dead'] = fuzz.trapmf(grade.universe, [0, 0, 25, 45])
    grade['Bad'] = fuzz.trimf(grade.universe, [25, 45, 65])
    grade['Normal'] = fuzz.trimf(grade.universe, [45, 65, 85])
    grade['Good'] = fuzz.trapmf(grade.universe, [65, 85, 100, 100])

# =============================================================================
#     grade['Class D'] = fuzz.trapmf(grade.universe, [0, 0, 25, 45])
#     grade['Class C'] = fuzz.trimf(grade.universe, [25, 45, 65])
#     grade['Class B'] = fuzz.trimf(grade.universe, [45, 65, 85])
#     grade['Class A'] = fuzz.trapmf(grade.universe, [65, 85, 100, 100])
# =============================================================================
    #view
# =============================================================================
#     distance.view()
#     angle.view()
#     grade.view()
# =============================================================================

    #%%
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

    grade_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4,
                                     rule5, rule6, rule7, rule8,
                                     rule9, rule10, rule11, rule12])
    grade_system = ctrl.ControlSystemSimulation(grade_ctrl)

    return grade, grade_system

#%%
def grade(grade = 'Score', grade_system = 'grade_system', input1 = 'input1', input2 = 'input2'):
    grade_system.input['Distance'] = input1
    grade_system.input['Angle'] = input2
    grade_system.compute()
    #view
# =============================================================================
#     grade.view(sim=grade_system)
#     print(grade_system.output['Score'])
# =============================================================================
    return grade_system.output['Score']
    
if __name__ =='__main__':
    main()