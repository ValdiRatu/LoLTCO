import sys

def getAPIKeyFromArgs():
  key = sys.argv[1]
  if key == "":
    print("No API key provided")
    raise Exception("No API key provided")
  return key