import uproot
import pandas as pd
import yaml


class dataset_cache:

    def __init__(self, path_dictionary, tree_name):
        self.opened_paths = {}
        self.__open_as_uproot(path_dictionary, tree_name)

    def __open_as_uproot(self, path_dictionary, tree_name):
        for key, path in path_dictionary.items():
            self.opened_paths[key] = uproot.open(path)[tree_name]

    def keys(self):
        return list(self.opened_paths.keys())

    def get_dataset(self, key):
        return self.opened_paths[key]

    def get_pandas(self, columns, key=None):
        if isinstance(key, list):
            return {k: v.arrays(columns, library='pd') for k, v in self.opened_paths.items() if k in key}
        if key is not None and not key == 'total':
            return self.opened_paths[key].arrays(columns, library='pd')
        elif key == 'total':
            pddf_list = [uprooted.arrays(columns, library='pd') for uprooted in self.opened_paths.values()]
            return pd.concat(pddf_list)
        elif key is None:
            return {k: v.arrays(columns, library='pd') for k, v in self.opened_paths.items()}
        else:
            raise KeyError('Unrecognised key, either undefined or not "total".')


class FeatherCache:

    def __init__(self, database_file):
        with open(database_file, "r") as stream:
            self.path_dict = yaml.safe_load(stream)


def read_parquet_dictionary(dic, *args, **kwargs):
    return {k:pd.read_parquet(*args, **kwargs) for k, p in dic.items()}
