import json
import os
class json_support:
    def __init__(self):
        pass
    def to_dict(self)->dict:
        pass

    def from_dict(self,d:dict,filefolder:str=""):
        pass
    def get_json(self):
        return json.dumps(self.to_dict(), indent=4)
    def to_json(self, filename):
        with open(filename,'w+') as f:
            f.write(self.get_json())

    def from_json(self, filename):
        if not os.path.exists(filename):
            raise IOError("File %s does not exist" % filename)
        file_folder,_ = os.path.split(filename)
        with open(filename) as f:
            self.from_dict(json.load(f),file_folder)

