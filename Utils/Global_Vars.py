import json

dictionary = None


class GlobalVars(object):
    def __init__(self, file_path=None):
        global dictionary
        if dictionary is not None:
            try:
                temp_path = "./global_vars.json"
                if file_path is not None:
                    temp_path = file_path
                with open(temp_path) as json_file:
                    dictionary = json.load(json_file)

            # Catch exceptions
            except Exception as e:
                print("Failed Loading Global Vars File.")
                raise ValueError('Check the Given File Path If It Was Given')

    def __getattr__(self, item):
        return dictionary[item]

    def __getitem__(self, item):
        return dictionary[item]
