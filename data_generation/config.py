from rules import *


#format: (rule, probability, name)
#Rule config for all 25 rules with their names and probabilities
#Add new rules you have implemented to this list,

rule_config = [
    (rule_1, 0.8, "rule_1"),
    (rule_2, 0.01, "rule_2"),
    (rule_3, 0.01, "rule_3"),
    (rule_4, 0.95, "rule_4"),
    (rule_5, 0.75, "rule_5"),
    (rule_6, 0.75, "rule_6"),
    (rule_7, 0.75, "rule_7"),
    (rule_8, 0.95, "rule_8"),
    (rule_9, 0.02, "rule_9"),
    (rule_10, 0.15, "rule_10"),
    (rule_11, 0.9, "rule_11"),
    (rule_12, 0.95, "rule_12"),
    (rule_13, 0.5, "rule_13"),
    (rule_14, 0.95, "rule_14"),
    (rule_15, 0.15, "rule_15"),
    (rule_16, 0.95, "rule_16"),
    (rule_17, 0.2, "rule_17"),
    (rule_18, 0.5, "rule_18"),
    (rule_19, 0.95, "rule_19"),
    (rule_20, 0.9, "rule_20"),
    (rule_21, 0.05, "rule_21"),
    (rule_22, 0.95, "rule_22"),
    (rule_23, 0.02, "rule_23"),
    (rule_24, 0.99, "rule_24"),
    (rule_25, 0.8, "rule_25"),
]

#Some rules add spaces to the sentence, some remove spaces from the sentence. To be able to match the indices, we must know which one is which
#For example, if a rule converts "firstsecond" to "first second", it is a space adder rule. If a rule converts "first second" to "firstsecond", it is a space remover rule.
#Update this lists if you add new rules


space_adding_rules = [rule_config[5][2],
                      rule_config[7][2],
                      rule_config[17][2],
                      rule_config[18][2],
                      rule_config[19][2],
                      rule_config[20][2],
                      rule_config[21][2]]

space_removing_rules = [rule_config[0][2], 
                        rule_config[3][2],
                        rule_config[4][2],
                        rule_config[6][2],
                        rule_config[9][2],
                        rule_config[10][2],
                        rule_config[16][2]]
