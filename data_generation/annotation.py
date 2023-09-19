from config import space_adding_rules, space_removing_rules, rule_config

class Annotation:
    def __init__(self, tag):
        tag = tag.split("|||")
        self.starting_index = int(tag[0].split(" ")[1])
        self.ending_index = int(tag[0].split(" ")[2])
        self.rule_name = tag[1]
        self.correction = tag[2]
        
        if self.rule_name in space_adding_rules:
            self.space_adding = True
            self.space_removing = False
        elif self.rule_name in space_removing_rules:
            self.space_removing = True
            self.space_adding = False
        else:
            self.space_removing = False
            self.space_adding = False
            
    def to_string(self):
        return f"A {self.starting_index} {self.ending_index}|||{self.rule_name}|||{self.correction}|||REQUIRED|||-NONE-|||0"
        
    def get_indices(self):
        return list(range(self.starting_index, self.ending_index)) 
    
    def get_tag_number(self):
        return int(self.rule_name.split("_")[-1])