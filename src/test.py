from datetime import datetime
from pathlib import Path
home = Path.home()
print(home)
filename = home.joinpath("air_pollution_estimation/src/t.txt")
print(filename)
myFile = open(f'{filename}', 'a')
myFile.write('\nAccessed on ' + str(datetime.now()))
myFile.close()
