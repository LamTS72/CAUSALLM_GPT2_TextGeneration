from configparser import ConfigParser

config_parser = ConfigParser()
config_parser.read("../config.ini")
parser = config_parser["API_KEY"]
